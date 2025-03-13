#!/usr/bin/env python3
"""Test script to demonstrate ML vs threshold-based detection using MQTT"""
import json
import logging
import time,datetime
from datetime import datetime,timezone
import numpy as np
from tabulate import tabulate
import random
from typing import Dict
import threading
import paho.mqtt.client as mqtt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def send_ml_data(scenario_type: str, ml_score: float):
    pass


class MLAdvantageDemo:
    def __init__(self, broker_address="192.168.1.143", broker_port=1883):
        self.broker_address = broker_address
        self.broker_port = broker_port
        self.device_id = None
        self.api_key = None
        self.results = []
        self.latest_metrics = None  # Store latest metrics received from collector
        self.training_progress = 0  # Track training progress
        # self.mqtt_client = mqtt.Client()
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        
        # MQTT Callbacks
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        
        # Topics
        self.metrics_topic = "metrics"
        self.status_topic = "status"
        self.control_topic = "control"
        self.anomalies_topic = "anomalies"

    def on_connect(self, client, userdata, flags, rc,properties=None):
        """Callback for MQTT connection"""
        if rc == 0:
            logging.info("Connected to MQTT broker")
            # Subscribe to relevant topics
            client.subscribe(f"{self.metrics_topic}/#")  # Subscribe to metrics topic
            client.subscribe(f"{self.status_topic}/#")   # Subscribe to status topic
            client.subscribe(f"{self.anomalies_topic}/#")  # Subscribe to anomalies topic
        else:
            logging.error(f"Failed to connect to MQTT broker, return code {rc}")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            logging.info(f"Received MQTT message on topic {msg.topic}: {payload}")
            
            # Handle status updates only for the target device
            if msg.topic == f"status/{self.device_id}":
                status = payload
                self.training_progress = status.get('training_data_size', 0)
                logging.info(f"Status update for device {self.device_id}: {status}")
                
                # Update is_trained only if the status is for the correct device
                if status.get('is_trained', False):
                    self.is_trained = True
                    logging.info("Model is trained. Proceeding with tests.")
                else:
                    logging.info("Model not yet trained. Waiting...")
            
            # Handle metrics and anomalies (existing logic)
            elif msg.topic == f"metrics/{self.device_id}":
                self.latest_metrics = payload.get('metrics', {})
                logging.info(f"Metrics received for device {self.device_id}: {self.latest_metrics}")
            
            elif msg.topic == f"anomalies/{self.device_id}":
                logging.info(f"Anomaly detected for device {self.device_id}: {payload}")
        
        except Exception as e:
            logging.error(f"Error processing MQTT message: {str(e)}")

    def check_thresholds(self, metrics):
        """Check if metrics exceed threshold values"""
        base_metrics = {
            'client_count': 20,
            'channel_utilization': 0.4,
            'retry_rate': 0.1,
            'noise_floor': -85,
            'tx_failed_rate': 0.02
        }
        
        return any([
            metrics.get('retry_rate', 0) > base_metrics['retry_rate'] * 1.5,
            metrics.get('channel_utilization', 0) > 0.8,
            metrics.get('noise_floor', -95) > base_metrics['noise_floor'] + 10,
            metrics.get('tx_failed_rate', 0) > base_metrics['tx_failed_rate'] * 2
        ])
    
    def get_status(self, device_id: str = None) -> Dict:
        """Get ML processing status for device via MQTT"""
        print("Get Status ML ADV TEST")
        if not device_id:
            device_id = self.device_id

        if not device_id:
            logging.error("Cannot get status: no device ID available")
            return {'error': 'No device ID available'}

        status_topic = f"status/{device_id}"
        status_received = None
        status_event = threading.Event()

        def on_status_message(client, userdata, msg):
            """Callback to handle status messages"""
            nonlocal status_received
            try:
                payload = json.loads(msg.payload.decode('utf-8'))
                logging.info(f"Received status for device {device_id}: {payload}")
                
                # Filter out request messages and only process actual status updates
                if 'is_trained' in payload or 'training_data_size' in payload:
                    status_received = payload
                    status_event.set()  # Signal that status has been received
                else:
                    logging.warning(f"Ignoring non-status message: {payload}")
            except Exception as e:
                logging.error(f"Error processing status message: {str(e)}")

        # Subscribe to the status topic
        self.mqtt_client.subscribe(status_topic)
        self.mqtt_client.message_callback_add(status_topic, on_status_message)

        # Publish a request for status
        self.mqtt_client.publish(status_topic, json.dumps({"request": "status"}))

        # Wait for the status response
        if status_event.wait(timeout=10):  # Wait up to 10 seconds for the status
            return status_received
        else:
            logging.error("Timeout waiting for status response")
            return {'error': 'Timeout waiting for status'}

    def wait_for_collector(self, timeout=30):
        """Wait for collector to become available"""
        start_time = time.time()
        logging.info("Waiting for collector to become available...")
        
        while time.time() - start_time < timeout:
            try:
                # Publish a test message to check connectivity
                test_topic = f"{self.status_topic}/{self.device_id}"
                self.mqtt_client.publish(test_topic, json.dumps({"status": "ping"}))
                
                # Wait for response
                time.sleep(2)
                if self.mqtt_client.is_connected():
                    logging.info("Collector is available")
                    return True
            except Exception as e:
                logging.warning(f"Connection failed: {str(e)}")
                time.sleep(2)
        
        logging.error("Collector not available after timeout")
        return False

    def setup(self):
        """Create test device and initialize collector"""
        try:
            # Connect to MQTT broker
            self.mqtt_client.connect(self.broker_address, self.broker_port, 60)
            self.mqtt_client.loop_start()
            
            # Wait for collector to be available
            if not self.wait_for_collector():
                raise Exception("Collector not available after timeout")
            
            # Create new device ID and API key
            self.device_id = "openwrt-23"
            self.api_key = "test-key-12345"  # Fixed API key for testing
            logging.info(f"Created device {self.device_id} with API key {self.api_key}")
            return True
        except Exception as e:
            logging.error(f"Setup failed: {str(e)}")
            return False

    def control_collector(self, command: str, scenario: str = "normal"):
        """Control the collector"""
        try:
            if command == "start":
                self.current_scenario = scenario
                logging.info(f"Starting collector with scenario: {scenario}")
            elif command == "stop":
                logging.info("Stopping collector")
        except Exception as e:
            logging.error(f"Error controlling collector: {str(e)}")

    def wait_for_training(self):
        """Wait for the model to be trained and start metrics collection"""
        try:
            # Start metrics collection in normal scenario
            self.control_collector("start", "normal")
            logging.info("Started metrics collection for training")
            while True:
                status = self.get_status(self.device_id)
                if status.get('is_trained', False):
                    logging.info("Model is trained. Proceeding with tests.")
                    return
                else:
                    training_samples = status.get('training_data_size', 0)
                    required_samples = 60
                    logging.info(f"Training progress: {training_samples}/{required_samples} samples")
                    if training_samples >= required_samples:
                        logging.warning("Collected sufficient samples, but model is not marked as trained.")
                    time.sleep(5)
        except Exception as e:
            logging.error(f"Error waiting for training: {str(e)}")
            time.sleep(5)

    def get_metrics(self):
        """Retrieve the latest metrics for the device via MQTT"""
        try:
            metrics_topic = f"metrics/{self.device_id}"
            metrics_received = None
            metrics_event = threading.Event()

            def on_metrics_message(client, userdata, msg):
                nonlocal metrics_received
                try:
                    payload = json.loads(msg.payload.decode('utf-8'))
                    logging.info(f"Received metrics for device {self.device_id}: {payload}")
                    metrics_received = payload.get('metrics', {})
                    metrics_event.set()
                except Exception as e:
                    logging.error(f"Error processing metrics message: {str(e)}")

            self.mqtt_client.subscribe(metrics_topic)
            self.mqtt_client.message_callback_add(metrics_topic, on_metrics_message)

            if metrics_event.wait(timeout=10):
                return [metrics_received]
            else:
                logging.error("Timeout waiting for metrics response")
                return []
        except Exception as e:
            logging.error(f"Error retrieving metrics: {str(e)}")
            return []
        

    def get_ml_score(self, metrics: Dict) -> float:
        """
        Retrieve the ML score (combined_score) from the anomaly detection results.
        """
        try:
            # Publish metrics to trigger anomaly detection
            self.send_metrics(metrics)
            time.sleep(2)  # Allow time for anomaly detection

            # Subscribe to anomalies topic to get the ML score
            anomalies_topic = f"anomalies/{self.device_id}"
            ml_score_received = None
            score_event = threading.Event()

            def on_anomaly_message(client, userdata, msg):
                """Callback to handle anomaly messages"""
                nonlocal ml_score_received
                try:
                    payload = json.loads(msg.payload.decode('utf-8'))
                    logging.info(f"Received anomaly for device {self.device_id}: {payload}")
                    
                    # Ensure payload is a list and extract the first element
                    if isinstance(payload, list) and len(payload) > 0:
                        anomaly = payload[0]
                        ml_score_received = anomaly.get('score', 0)  # Extract combined_score
                        score_event.set()  # Signal that score has been received
                    else:
                        logging.error(f"Unexpected payload format: {payload}")
                except Exception as e:
                    logging.error(f"Error processing anomaly message: {str(e)}")

            # Subscribe to anomalies topic
            self.mqtt_client.subscribe(anomalies_topic)
            self.mqtt_client.message_callback_add(anomalies_topic, on_anomaly_message)

            # Wait for the ML score response
            if score_event.wait(timeout=10):  # Wait up to 10 seconds for the score
                return ml_score_received
            else:
                logging.error("Timeout waiting for ML score")
                return 0.0
        except Exception as e:
            logging.error(f"Error retrieving ML score: {str(e)}")
        return 0.0
        
    def send_metrics(self, metrics: Dict):
        """Publish metrics via MQTT"""
        payload = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'device_id': self.device_id,
            'metrics': metrics
        }
        result, mid = self.mqtt_client.publish(self.metrics_topic, json.dumps(payload))
        if result != mqtt.MQTT_ERR_SUCCESS:
            logging.error(f"Failed to publish metrics: {mqtt.error_string(result)}")
    
    def send_control_message(self, scenario, test_mode=True):
        """Send a control message to change the scenario."""
        control_message = {
            "command": "start",
            "scenario": scenario,
            "test_mode": test_mode
        }
        topic = f"control/{self.device_id}"  # Use dynamic topic based on device ID
        result, mid = self.mqtt_client.publish(topic, json.dumps(control_message))
        if result != mqtt.MQTT_ERR_SUCCESS:
            logging.error(f"Failed to publish control message: {mqtt.error_string(result)}")
        else:
            logging.info(f"Published control message to {topic}: {control_message}")

    def demonstrate_advantage(self):
        """Run the ML advantage demonstration"""
        if not self.api_key:
            logging.error("Cannot demonstrate: no API key available")
            return
        
        logging.info("=== Starting ML Advantage Demonstration ===")
        
        # Phase 1: Training
        logging.info("Phase 1: Training ML model on normal patterns")
        self.wait_for_training()  # Wait for the model to be trained
        
        # Phase 2: Demonstration
        logging.info("Phase 2: Demonstrating detection capabilities")
        headers = ["Time", "Scenario", "Metrics", "Threshold Alert", "ML Score", "ML Alert"]
        results = []
        test_scenarios = [("Normal Operation", "normal"), ("Subtle Combined Anomaly", "subtle"), ("Obvious Problem", "obvious")]
        
        for scenario_name, scenario_type in test_scenarios:
            logging.info(f"Testing scenario: {scenario_name}")
            try:
                # Send control message to start collector with scenario
                self.send_control_message(scenario_type)
                
                # Wait for confirmation that the scenario has changed
                self.wait_for_scenario_change(scenario_type)
                
                # Get latest metrics and ML scores
                metrics_data = self.get_metrics()
                if metrics_data and len(metrics_data) > 0:
                    metrics = metrics_data[0]
                    ml_score = self.get_ml_score(metrics)
                    logging.info(f"ML score -> {ml_score}, scenario -> {scenario_type}")
                    
                    # Format metrics for display
                    metrics_display = json.dumps(metrics, indent=2) if metrics else "No metrics"
                    
                    # Add result
                    results.append([
                        datetime.now().strftime("%H:%M:%S"),
                        scenario_name,
                        metrics_display,
                        "✓" if self.check_thresholds(metrics) else "✗",
                        f"{ml_score:.3f}",
                        "✓" if ml_score > 1.0 else "✗"
                    ])
                
                # Stop collector
                self.send_control_message("normal")  # Reset to normal
                logging.info(f"Sent control message to stop scenario: {scenario_type}")
            except Exception as e:
                logging.error(f"Error during scenario {scenario_name}: {str(e)}")
        
        # Print results
        logging.info("Results:")
        logging.info(tabulate(results, headers=headers, tablefmt="grid"))
        logging.info("Key Findings:")
        logging.info("1. Threshold-based detection misses subtle combined anomalies")
        logging.info("2. ML detection identifies correlated metric changes")
        logging.info("3. Both systems detect obvious problems")

    def wait_for_scenario_change(self, expected_scenario, timeout=10):
        """Wait for the scenario to change to the expected value."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_status(self.device_id)
            current_scenario = status.get("current_scenario", "unknown")
            if current_scenario == expected_scenario:
                logging.info(f"Scenario changed to {expected_scenario}")
                return True
            time.sleep(1)
        logging.warning(f"Timeout waiting for scenario to change to {expected_scenario}")
        return False

def main():
    demo = MLAdvantageDemo()
    
    if not demo.setup():
        logging.error("Setup failed, exiting")
        return
    
    while True:
        try: 
            demo.demonstrate_advantage()
        except Exception as e:
            logging.error(f"Error during demonstration: {str(e)}")
        
        sleep_time = random.randint(1, 4)
        logging.info(f"----- Waiting for {sleep_time} seconds -----")
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()
