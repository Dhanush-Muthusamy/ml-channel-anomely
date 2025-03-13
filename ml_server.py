# ml_server.py       
import json
import logging
from datetime import datetime, timezone, UTC
import threading
from collections import defaultdict
import os
import time
import paho.mqtt.client as mqtt
from typing import Dict
from ml_components import MLAnomalyDetector

# Directory setup
BASE_DIR = os.path.expanduser('~/ml-channel-mqtt/server')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'server.log')

# Create directories if they don't exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Initialize MQTT client
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

class MLProcessor:
    """Manages ML processing for multiple devices"""
    def __init__(self):
        self.detectors = {}
        self.lock = threading.Lock()
        self.training_data = defaultdict(list)
        self.save_interval = 3600  # Save models every hour
        self.last_save = defaultdict(float)
    
    def get_detector(self, device_id: str) -> MLAnomalyDetector:
        """Get or create an ML detector for a device"""
        with self.lock:
            if device_id not in self.detectors:
                model_path = os.path.join(MODEL_DIR, device_id)
                os.makedirs(model_path, exist_ok=True)
                logging.info(f"Creating detector with path: {model_path}")
                config = {
                    'model_path': model_path,
                    'window_size': 60,
                    'n_components': 3,
                    'n_clusters': 5,
                    'contamination': 0.1
                }
                self.detectors[device_id] = MLAnomalyDetector(config)
                self.detectors[device_id].load_models()  # Try to load existing models
            
            return self.detectors[device_id]
    
    def process_metrics(self, device_id: str, metrics: Dict) -> Dict:
        """Process metrics for a device"""
        detector = self.get_detector(device_id)
        
        # Update training data
        self.training_data[device_id].append(metrics)
        if len(self.training_data[device_id]) > 1440:  # Keep 24 hours of data
            self.training_data[device_id].pop(0)
        
        # Check if we should start training
        training_size = len(self.training_data[device_id])
        logging.info(f"Training data size for device {device_id}: {training_size}")
        
        if not detector.is_trained:
            if training_size >= 60:  # Require 60 samples for training
                logging.info(f"Starting initial training for device {device_id}")
                success = detector.train(self.training_data[device_id])
                if success:
                    detector.is_trained = True  # Mark the model as trained
                    logging.info("Initial training completed successfully")
                    # Publish status update
                    status = self.get_status(device_id)
                    mqtt_client.publish(f"status/{device_id}", json.dumps(status))
                    return {
                        'status': 'training_completed',
                        'training_metrics': {
                            'training_samples': training_size,
                            'required_samples': 60
                        }
                    }
                else:
                    logging.error("Initial training failed")
            else:
                logging.info(f"Collecting training data: {training_size}/60 samples")
                return {
                    'status': 'collecting_training_data',
                    'training_metrics': {
                        'training_samples': training_size,
                        'required_samples': 60
                    }
                }
        else:
            # Log training metrics even when already trained
            training_metrics = detector.get_training_metrics()
            logging.info(f"Training metrics for trained model: {training_metrics}")
        
        # Process metrics only if trained
        if detector.is_trained:
            result = detector.update(metrics)
        else:
            result = {
                'anomalies': [],
                'scores': {},
                'error': 'Training incomplete'
            }
        
        # Save models periodically
        current_time = time.time()
        if current_time - self.last_save[device_id] > self.save_interval:
            detector.save_models()
            self.last_save[device_id] = current_time
        
        return result
    
    def get_status(self, device_id: str) -> dict:
        """Get ML processing status for a device"""
        if device_id in self.detectors:
            detector = self.detectors[device_id]
            training_size = len(self.training_data[device_id])
            
            status = {
                'is_trained': detector.is_trained,
                'training_data_size': training_size,
                'last_save': datetime.fromtimestamp(
                    self.last_save[device_id]
                ).isoformat() if self.last_save[device_id] > 0 else None
            }
            status.update(detector.get_training_metrics())
            logging.info(f"Status for device {device_id}: {status}")
            return status
        
        return {'error': 'Device not found'}

# Initialize ML processor
ml_processor = MLProcessor()

def on_connect(client, userdata, flags, rc, properties=None):
    """Callback for MQTT connection"""
    if rc == 0:
        logging.info("Connected to MQTT broker")
        client.subscribe("metrics/#")  # Subscribe to metrics topic
        client.subscribe("status/#")   # Subscribe to status topic
    else:
        logging.error(f"Failed to connect to MQTT broker, return code {rc}")

def on_message(client, userdata, msg):
    """Callback for MQTT messages"""
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        logging.info(f"Received MQTT message on topic {msg.topic}: {payload}")
        
        # Handle status requests
        if msg.topic.startswith("status/"):
            device_id = msg.topic.split('/')[-1]
            if payload.get("request") == "status":
                status = ml_processor.get_status(device_id)
                client.publish(msg.topic, json.dumps(status))  # Respond to the same topic
                logging.info(f"Responded to status request for device {device_id}")
            return
        
        # Handle metrics messages (existing logic)
        device_id = msg.topic.split('/')[-1]
        if not device_id:
            logging.warning("No device ID found in topic")
            return
        
        # Validate payload
        if not all(k in payload for k in ['timestamp', 'metrics']):
            logging.warning(f"Missing required fields in payload: {payload}")
            return
        
        # Process metrics
        metrics = payload['metrics']
        result = ml_processor.process_metrics(device_id, metrics)
        logging.info(f"ML processing result for device {device_id}: {result}")
        
        # Publish anomalies
        if result.get('anomalies'):
            anomaly_topic = f"anomalies/{device_id}"
            client.publish(anomaly_topic, json.dumps(result['anomalies']))
        
        # Publish status updates (existing logic)
        status_topic = f"status/{device_id}"
        status = ml_processor.get_status(device_id)
        client.publish(status_topic, json.dumps(status))
    
    except Exception as e:
        logging.error(f"Error processing MQTT message: {str(e)}")

def main():
    # Configure MQTT client
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    
    # Connect to MQTT broker
    broker_address = "192.168.1.143"
    broker_port = 1883
    try:
        mqtt_client.connect(broker_address, broker_port, 60)
        mqtt_client.loop_start()
        logging.info(f"Connected to MQTT broker at {broker_address}:{broker_port}")
    except Exception as e:
        logging.error(f"Failed to connect to MQTT broker: {str(e)}")
        return
    
    # Keep the server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down server...")
        mqtt_client.loop_stop()
        mqtt_client.disconnect()

if __name__ == "__main__":
    main()
