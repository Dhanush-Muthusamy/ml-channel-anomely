#!/usr/bin/env python3
"""ML Components for Network Monitoring"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import MiniBatchKMeans
import json
import logging
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from collections import deque
import os
import joblib
from datetime import datetime, timedelta, timezone, UTC


class LSTMPredictor(nn.Module):
    """LSTM network for sequence prediction"""
    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

class EnhancedFeatureExtractor:
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.metric_history = {}
        self.derived_features = {}
        self.feature_names = None  # Track feature names
    
    def add_metrics(self, metrics: Dict[str, float]):
        """Store raw metrics and compute derived features"""
        # Debug log incoming metrics
        logging.info(f"Adding metrics: {metrics}")
        
        # Store raw metrics
        for name, value in metrics.items():
            if name not in self.metric_history:
                self.metric_history[name] = deque(maxlen=self.window_size)
            self.metric_history[name].append(float(value))
        
        # Update derived features
        self._update_derived_features()
        
        # Debug log final feature vector
        vector, names = self.get_feature_vector()
        logging.info(f"Generated features: {dict(zip(names, vector))}")

    def _update_derived_features(self):
        """Update derived features from raw metrics"""
        self.derived_features = {}  # Reset derived features
        
        if not all(len(hist) > 0 for hist in self.metric_history.values()):
            return
        
        # Get latest values
        latest = {k: float(list(v)[-1]) for k, v in self.metric_history.items()}
        logging.info(f"Latest values for derived features: {latest}")
        
        # Calculate rate of change features
        for name, history in self.metric_history.items():
            if len(history) >= 2:
                rate = float((history[-1] - history[-2]) / 1.0)  # per minute
                self.derived_features[f'{name}_rate'] = rate
        
        # Network efficiency features
        if 'client_count' in latest and latest['client_count'] > 0:
            self.derived_features['per_client_utilization'] = float(
                latest.get('channel_utilization', 0) / latest['client_count']
            )
            self.derived_features['per_client_retries'] = float(
                latest.get('retry_rate', 0) / latest['client_count']
            )
        
        # Signal quality index
        self.derived_features['signal_quality'] = float(
            (100 + latest.get('noise_floor', -95)) / 100 +
            (1 - latest.get('retry_rate', 0)) +
            (1 - latest.get('channel_utilization', 0))
        ) / 3
        
        # Network load index
        self.derived_features['network_load'] = float(
            latest.get('channel_utilization', 0) * 0.5 +
            (latest.get('client_count', 0) / 50) * 0.3 +
            latest.get('retry_rate', 0) * 0.2
        )
        
        logging.info(f"Updated derived features: {self.derived_features}")

    def get_feature_vector(self) -> Tuple[np.ndarray, List[str]]:
        """Get feature vector and names"""
        # Combine raw metrics and derived features
        features = {}
        
        # Add raw metrics
        features.update({k: float(list(v)[-1]) for k, v in self.metric_history.items()})
        
        # Add derived features
        features.update(self.derived_features)
        
        # Add temporal features
        hour = datetime.now().hour
        day = datetime.now().weekday()
        features.update({
            'hour_sin': float(np.sin(2 * np.pi * hour / 24)),
            'hour_cos': float(np.cos(2 * np.pi * hour / 24)),
            'day_sin': float(np.sin(2 * np.pi * day / 7)),
            'day_cos': float(np.cos(2 * np.pi * day / 7))
        })
        
        # If feature names not set, initialize them
        if self.feature_names is None:
            self.feature_names = list(features.keys())
            logging.info(f"Initialized feature names: {self.feature_names}")
        
        # Ensure consistent feature order
        feature_vector = []
        for name in self.feature_names:
            value = features.get(name, 0.0)
            feature_vector.append(float(value))
            
        logging.info(f"Generated feature vector with {len(feature_vector)} features")
        
        return np.array(feature_vector), self.feature_names


class DetailedAnomalyAnalyzer:
    def __init__(self, config=None):
        self.config = config or {}
        # Default severity levels if not provided in config
        self.severity_levels = self.config.get('severity_levels', {
            'low': (1.0, 3.0),
            'medium': (3.0, 7.0),
            'high': (7.0, float('inf'))
        })

    def _determine_severity(self, score: float) -> Dict:
        """Map ML score to severity level"""
        try:
            for level, (min_val, max_val) in self.severity_levels.items():
                if min_val <= score < max_val:
                    return {
                        'level': level,
                        'score': score,
                        'description': self._get_severity_description(level, score)
                    }
            return {
                'level': 'normal',
                'score': score,
                'description': 'Operating within normal parameters'
            }
        except Exception as e:
            logging.error(f"Error determining severity: {str(e)}")
            return {
                'level': 'unknown',
                'score': score,
                'description': 'Unable to determine severity'
            }

    def analyze_anomaly(self, combined_score: float, features: List[Dict], scores: Dict) -> Dict:
        """Analyze anomaly and provide detailed report"""
        try:
            severity = self._determine_severity(combined_score)
            
            analysis = {
                'severity': severity,
                'primary_factors': self._analyze_primary_factors(features),
                'technical_details': self._generate_technical_details(scores, features),
                'recommendations': self._generate_recommendations(features, severity)
            }
            
            return analysis
        except Exception as e:
            logging.error(f"Error in anomaly analysis: {str(e)}")
            return {
                'severity': {
                    'level': 'unknown',
                    'score': combined_score,
                    'description': 'Analysis error'
                },
                'error': str(e)
            }

    def _analyze_primary_factors(self, features: List[Dict]) -> List[Dict]:
        """Analyze the primary contributing factors"""
        try:
            return [{
                'metric': f['feature'],
                'deviation': f['deviation'],
                'impact': 'high' if f['deviation'] > 5 else 'medium' if f['deviation'] > 2 else 'low'
            } for f in sorted(features, key=lambda x: x['deviation'], reverse=True)[:3]]
        except Exception as e:
            logging.error(f"Error analyzing primary factors: {str(e)}")
            return []

    def _analyze_primary_factors(self, features: List[Dict]) -> List[Dict]:
        """Analyze the primary contributing factors"""
        return [{
            'metric': f['feature'],
            'deviation': f['deviation'],
            'impact': 'high' if f['deviation'] > 5 else 'medium' if f['deviation'] > 2 else 'low'
        } for f in sorted(features, key=lambda x: x['deviation'], reverse=True)[:3]]

    def _generate_technical_details(self, scores: Dict, features: List[Dict]) -> Dict:
        """Generate detailed technical analysis"""
        return {
            'model_scores': {
                name: {
                    'score': score,
                    'interpretation': self._interpret_model_score(name, score)
                }
                for name, score in scores.items()
            },
            'feature_contributions': self._analyze_feature_contributions(features)
        }

    def _generate_recommendations(self, features: List[Dict], severity: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if severity['level'] == 'high':
            recommendations.append("Immediate investigation required")
        elif severity['level'] == 'medium':
            recommendations.append("Schedule investigation within 24 hours")
        else:
            recommendations.append("Monitor for continued deviation")

        # Add specific recommendations based on features
        for feature in features:
            if feature['deviation'] > 5:
                recommendations.append(f"Investigate {feature['feature']}: {feature['deviation']}x normal variation")

        return recommendations

    def _get_severity_description(self, level: str, score: float) -> str:
        """Get detailed severity description"""
        descriptions = {
            'low': "Minor anomaly detected - monitoring recommended",
            'medium': "Significant deviation from normal patterns",
            'high': "Critical anomaly requiring immediate attention"
        }
        return f"{descriptions.get(level, 'Unknown')} (Score: {score:.2f})"

    def _interpret_model_score(self, model: str, score: float) -> str:
        """Interpret individual model scores"""
        if model == 'isolation_forest':
            return f"{'High' if score > 0.6 else 'Medium' if score > 0.4 else 'Low'} isolation score"
        elif model == 'lof':
            return f"{'High' if score > 2 else 'Medium' if score > 1 else 'Low'} local outlier factor"
        elif model == 'lstm':
            return f"{'High' if score > 10 else 'Medium' if score > 5 else 'Low'} prediction error"
        return f"Score: {score:.2f}"

    def _analyze_feature_contributions(self, features: List[Dict]) -> List[Dict]:
        """Analyze how each feature contributes to the anomaly"""
        return [{
            'feature': f['feature'],
            'deviation': f['deviation'],
            'significance': 'High' if f['deviation'] > 5 else 'Medium' if f['deviation'] > 2 else 'Low',
            'technical_details': {
                'z_score': f['deviation'],
                'statistical_significance': 'p < 0.001' if f['deviation'] > 5 else 'p < 0.01' if f['deviation'] > 2 else 'p < 0.05'
            }
        } for f in features]
        
class MLAnomalyDetector:
    
    def __init__(self, config: Dict = None):
        base_path = os.path.expanduser('~/ml-channel-mqtt/server/models')
        self.config = config or {
            'window_size': 60,
            'n_components': 3,
            'n_clusters': 5,
            'contamination': 0.1,
            'model_path': base_path
        }
        
        # Create model directory
        os.makedirs(self.config['model_path'], exist_ok=True)
        
        # Initialize components     
        self.feature_extractor = EnhancedFeatureExtractor(self.config['window_size'])
        self.scaler = StandardScaler()
        self.pca = IncrementalPCA(n_components=self.config['n_components'])
        self.isolation_forest = IsolationForest(
            n_estimators=50,
            contamination=self.config['contamination']
        )
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.config['contamination'],
            novelty=True
        )
        self.clustering = MiniBatchKMeans(
            n_clusters=self.config['n_clusters'],
            batch_size=100
        )
        
        # Initialize LSTM parameters
        self.sequence_length = 10
        self.lstm_hidden_size = 32
        self.sequence_buffer = deque(maxlen=self.sequence_length)
        self.lstm = None
        self.input_size = None
        
        self.is_trained = False
        self.feature_names = []
        # Initialize n_features
        self.n_features = None
        
        self.training_metrics = {
            'training_accuracy': 0.0,
            'false_positive_rate': 0.0,
            'training_samples': 0
        }
        self.analyzer = DetailedAnomalyAnalyzer(config)

    def _train_lstm(self, X_scaled: np.ndarray):
        """Train LSTM model on scaled data"""
        try:
            if len(X_scaled) < self.sequence_length + 1:
                logging.warning("Not enough data to train LSTM")
                return

            # Convert data to sequences
            sequences = []
            targets = []
            
            # Create sequences and targets
            for i in range(len(X_scaled) - self.sequence_length):
                seq = X_scaled[i:i+self.sequence_length]
                target = X_scaled[i+self.sequence_length]
                sequences.append(seq)
                targets.append(target)
            
            if not sequences:
                logging.warning("No sequences generated for LSTM training")
                return
                
            # Convert to numpy arrays first
            X_seq = np.array(sequences)
            y_seq = np.array(targets)
            
            # Then convert to torch tensors
            X_seq = torch.FloatTensor(X_seq)
            y_seq = torch.FloatTensor(y_seq)
            
            # Initialize LSTM if needed
            input_size = X_seq.shape[2]  # Get input size from data
            if self.lstm is None:
                self.lstm = LSTMPredictor(input_size, self.lstm_hidden_size)
            
            # Train the model
            self.lstm.train()
            optimizer = torch.optim.Adam(self.lstm.parameters())
            criterion = nn.MSELoss()
            
            # Training loop
            for epoch in range(50):
                optimizer.zero_grad()
                output = self.lstm(X_seq)
                loss = criterion(output, y_seq)
                loss.backward()
                optimizer.step()
                
            logging.info("LSTM training completed successfully")
            
        except Exception as e:
            logging.error(f"Error in LSTM training: {str(e)}")
            self.lstm = None  # Reset LSTM on error

    def train(self, initial_data: List[Dict[str, float]]) -> bool:
        """Train all ML models with initial data"""
        try:
            if len(initial_data) < self.config['window_size']:
                logging.warning(f"Not enough training data: {len(initial_data)} < {self.config['window_size']}")
                return False
            
            # Process all data through feature extractor
            feature_vectors = []
            for metrics in initial_data:
                self.feature_extractor.add_metrics(metrics)
                feature_vector, self.feature_names = self.feature_extractor.get_feature_vector()
                feature_vectors.append(feature_vector)
            
            # Convert to numpy array
            X = np.array(feature_vectors, dtype=np.float64)
            logging.info(f"Training data shape: {X.shape}")
            
            # Store feature dimension
            self.n_features = X.shape[1]
            logging.info(f"Feature dimension: {self.n_features}")
            logging.info(f"Feature names: {self.feature_names}")
            
            # Fit models
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            self.pca.fit(X_scaled)
            X_pca = self.pca.transform(X_scaled)
            
            self.isolation_forest.fit(X_pca)
            self.lof.fit(X_pca)
            self.clustering.fit(X_pca)
            
            # Train LSTM
            self._train_lstm(X_scaled)
            
            # Initialize training metrics
            predictions = self.isolation_forest.predict(X_pca)
            anomalies = sum(predictions == -1)
            total = len(predictions)
            
            self.training_metrics['training_samples'] = total
            self.training_metrics['training_accuracy'] = 1.0 - (anomalies / total if total > 0 else 0)
            self.training_metrics['false_positive_rate'] = anomalies / total if total > 0 else 0
            
            self.is_trained = True
            logging.info("All models trained successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            return False
        
    def update(self, metrics: Dict[str, float]) -> Dict:
        """Process new metrics and detect anomalies"""
        try:
            # Update feature extractor
            self.feature_extractor.add_metrics(metrics)
            feature_vector, feature_names = self.feature_extractor.get_feature_vector()
            
            logging.info(f"Processing metrics with {len(feature_vector)} features")
            
            # Initialize n_features if not set
            if self.n_features is None:
                self.n_features = len(feature_vector)
                self.feature_names = feature_names
                logging.info(f"Initialized with {self.n_features} features: {feature_names}")
            
            # Detect anomalies if trained
            if self.is_trained:
                result = self.detect_anomalies(feature_vector)
                return result
            else:
                logging.warning("Models not yet trained")
                return {'anomalies': [], 'scores': {}, 'error': 'Models not trained'}
                
        except Exception as e:
            logging.error(f"Error in update: {str(e)}")
            return {'anomalies': [], 'scores': {}, 'error': str(e)}
        
        # # Initialize LSTM parameters
        # self.sequence_length = 10
        # self.lstm_hidden_size = 32
        # self.sequence_buffer = deque(maxlen=self.sequence_length)
        # self.lstm = None  # Will be initialized during training
        # self.input_size = None  # Will be set during training
        
        # self.is_trained = False
        # self.feature_names = []
        # self.n_features = None  # Track expected feature dimension
        
        # self.training_metrics = {
        #     'training_accuracy': 0.0,
        #     'false_positive_rate': 0.0,
        #     'training_samples': 0
        # }
        # self.analyzer = DetailedAnomalyAnalyzer(config)
    
    def _update_training_metrics(self, feature_vector: np.ndarray):
        X_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        X_pca = self.pca.transform(X_scaled)
        prediction = self.isolation_forest.predict(X_pca)[0]
        
        # Update rolling accuracy
        alpha = 0.1  # Rolling average factor
        accuracy = 1.0 if prediction == 1 else 0.0
        self.training_metrics['training_accuracy'] = (
            (1 - alpha) * self.training_metrics['training_accuracy'] +
            alpha * accuracy
        )
        
        # Update false positive rate
        self.training_metrics['false_positive_rate'] = (
            (1 - alpha) * self.training_metrics['false_positive_rate'] +
            alpha * (1.0 if prediction == -1 else 0.0)
        )
        
        self.training_metrics['training_samples'] += 1

    
    def detect_anomalies(self, feature_vector: np.ndarray) -> Dict:
        """Detect anomalies with detailed analysis"""
        try:
            # Verify feature dimension
            if len(feature_vector) != self.n_features:
                raise ValueError(f"Expected {self.n_features} features, got {len(feature_vector)}")
                
            X_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
            X_pca = self.pca.transform(X_scaled)
            
            scores = {
                'isolation_forest': -self.isolation_forest.score_samples(X_pca)[0],
                'lof': -self.lof.score_samples(X_pca)[0],
                'cluster_distance': self.clustering.transform(X_pca).min()
            }
            
            if len(self.sequence_buffer) == self.sequence_length:
                scores['lstm'] = self._get_lstm_score(X_scaled)
            
            combined_score = np.mean(list(scores.values()))
            
            anomalies = []
            if combined_score > self.config['contamination']:
                features = self._identify_anomalous_features(X_scaled[0])
                
                try:
                    analysis = self.analyzer.analyze_anomaly(
                        combined_score=combined_score,
                        features=features,
                        scores=scores
                    )
                except Exception as e:
                    logging.warning(f"Error in detailed analysis: {str(e)}")
                    analysis = {
                        'error': str(e),
                        'severity': {
                            'level': 'unknown',
                            'score': combined_score
                        }
                    }
                
                anomalies.append({
                    'score': combined_score,
                    'features': features,
                    'analysis': analysis,
                    'timestamp': datetime.now(UTC).isoformat()
                })
            
            return {
                'anomalies': anomalies,
                'scores': scores,
                'combined_score': combined_score,
                'training_metrics': self.training_metrics
            }
            
        except Exception as e:
            logging.error(f"Error in anomaly detection: {str(e)}")
            return {
                'anomalies': [],
                'scores': {},
                'combined_score': 0,
                'training_metrics': self.training_metrics,
                'error': str(e)
            }
                
    def _get_lstm_score(self, X_scaled: np.ndarray) -> float:
        sequence = torch.FloatTensor(list(self.sequence_buffer))
        sequence = sequence.unsqueeze(0)
        
        self.lstm.eval()
        with torch.no_grad():
            prediction = self.lstm(sequence)
        
        actual = torch.FloatTensor(X_scaled)
        error = torch.mean((prediction - actual) ** 2).item()
        
        return error
    
    def _identify_anomalous_features(self, scaled_features: np.ndarray) -> List[Dict]:
        contributions = []
        for idx, (feature, value) in enumerate(zip(self.feature_names, scaled_features)):
            if abs(value) > 2:  # More than 2 standard deviations
                contributions.append({
                    'feature': feature,
                    'deviation': abs(value)
                })
        return sorted(contributions, key=lambda x: x['deviation'], reverse=True)
    
    def get_training_metrics(self) -> Dict:
        return self.training_metrics
    
    def save_models(self) -> bool:
        try:
            model_path = self.config['model_path']
            joblib.dump(self.scaler, os.path.join(model_path, 'scaler.joblib'))
            joblib.dump(self.pca, os.path.join(model_path, 'pca.joblib'))
            joblib.dump(self.isolation_forest, os.path.join(model_path, 'iforest.joblib'))
            joblib.dump(self.lof, os.path.join(model_path, 'lof.joblib'))
            joblib.dump(self.clustering, os.path.join(model_path, 'clustering.joblib'))
            
            if self.lstm is not None:
                torch.save(self.lstm.state_dict(),
                         os.path.join(model_path, 'lstm.pth'))
            
            with open(os.path.join(model_path, 'feature_names.json'), 'w') as f:
                json.dump(self.feature_names, f)
            
            with open(os.path.join(model_path, 'training_metrics.json'), 'w') as f:
                json.dump(self.training_metrics, f)
            
            return True
        except Exception as e:
            logging.error(f"Error saving models: {str(e)}")
            return False
    
    def load_models(self) -> bool:
        try:
            model_path = self.config['model_path']
            self.scaler = joblib.load(os.path.join(model_path, 'scaler.joblib'))
            self.pca = joblib.load(os.path.join(model_path, 'pca.joblib'))
            self.isolation_forest = joblib.load(os.path.join(model_path, 'iforest.joblib'))
            self.lof = joblib.load(os.path.join(model_path, 'lof.joblib'))
            self.clustering = joblib.load(os.path.join(model_path, 'clustering.joblib'))
            
            with open(os.path.join(model_path, 'feature_names.json'), 'r') as f:
                self.feature_names = json.load(f)
            
            lstm_path = os.path.join(model_path, 'lstm.pth')
            if os.path.exists(lstm_path):
                if self.lstm is None:
                    feature_size = len(self.feature_names)
                    self.lstm = LSTMPredictor(feature_size)
                self.lstm.load_state_dict(torch.load(lstm_path))
            
            with open(os.path.join(model_path, 'training_metrics.json'), 'r') as f:
                self.training_metrics = json.load(f)
            
            self.is_trained = True
            return True
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            return False
