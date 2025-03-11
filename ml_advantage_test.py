#!/usr/bin/env python3
"""Test script to demonstrate ML vs threshold-based detection using MQTT"""
import json
import logging
import time,datetime
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
        """Callback for MQTT messages"""
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            logging.info(f"Received MQTT message on topic {msg.topic}: {payload}")
            
            # Handle metrics updates
            if msg.topic.startswith(self.metrics_topic):
                device_id = msg.topic.split('/')[-1]
                self.latest_metrics = payload.get('metrics', {})
                logging.info(f"Metrics received for device {device_id}: {self.latest_metrics}")
            
            # Handle status updates (FIX: Use root payload instead of 'ml_status')
            elif msg.topic.startswith(self.status_topic):
                device_id = msg.topic.split('/')[-1]
                status = payload  # Directly use the payload
                self.training_progress = status.get('training_data_size', 0)
                logging.info(f"Status update for device {device_id}: {status}")
            
            # Handle anomalies
            elif msg.topic.startswith(self.anomalies_topic):
                device_id = msg.topic.split('/')[-1]
                logging.info(f"Anomaly detected for device {device_id}: {payload}")
        
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
        """Control the OpenWRT collector via MQTT"""
        try:
            payload = {
                "command": command,
                "scenario": scenario,
                "test_mode": True
            }
            
            logging.info(f"Sending control command: {command}, scenario: {scenario}")
            print("=========>>>self.control_topic====>>>",self.control_topic,"payload=====>>>>",payload)
            self.mqtt_client.publish(self.control_topic, json.dumps(payload))
            
        except Exception as e:
            logging.error(f"Error controlling collector: {str(e)}")
            raise

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
            # Subscribe to the metrics topic temporarily
            metrics_topic = f"metrics/{self.device_id}"
            metrics_received = None
            metrics_event = threading.Event()

            def on_metrics_message(client, userdata, msg):
                """Callback to handle metrics messages"""
                nonlocal metrics_received
                try:
                    payload = json.loads(msg.payload.decode('utf-8'))
                    logging.info(f"Received metrics for device {self.device_id}: {payload}")
                    metrics_received = payload.get('metrics', {})
                    metrics_event.set()  # Signal that metrics have been received
                except Exception as e:
                    logging.error(f"Error processing metrics message: {str(e)}")

            # Subscribe to the metrics topic
            self.mqtt_client.subscribe(metrics_topic)
            self.mqtt_client.message_callback_add(metrics_topic, on_metrics_message)

            # Wait for the metrics response
            if metrics_event.wait(timeout=10):  # Wait up to 10 seconds for metrics
                return [metrics_received]  # Return as a list for compatibility
            else:
                logging.error("Timeout waiting for metrics response")
                return []
        except Exception as e:
            logging.error(f"Error retrieving metrics: {str(e)}")
            return []

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
                self.control_collector("start", scenario_type)
                logging.info(f"Sent control message to start scenario: {scenario_type}")
                time.sleep(10)  # Wait for metrics to flow
                
                # Get latest metrics and calculate ML score
                metrics_data = self.get_metrics()
                if metrics_data and len(metrics_data) > 0:
                    metrics = metrics_data[0]
                    ml_score = self.get_ml_score(metrics)
                    logging.info(f"ML score -> {ml_score}, scenario -> {scenario_type}")
                    
                    # Add result
                    results.append([
                        time.strftime("%H:%M:%S"),
                        scenario_name,
                        json.dumps(metrics, indent=2),
                        "✓" if self.check_thresholds(metrics) else "✗",
                        f"{ml_score:.3f}",
                        "✓" if ml_score > 1.0 else "✗"
                    ])
                
                # Stop collector
                self.control_collector("stop")
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