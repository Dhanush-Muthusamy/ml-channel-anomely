#!/usr/bin/env python3
"""Integrated OpenWRT Metrics Collector with MQTT Control API"""

import json
import logging
import time
import random
import threading
from datetime import datetime
import numpy as np
from typing import Dict
import paho.mqtt.client as mqtt
import argparse
import subprocess
import signal
import sys

# Declare collector as a global variable
collector = None

def signal_handler(sig, frame):
    logging.info("Received termination signal. Stopping collector...")
    if collector:
        collector.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class OpenWRTCollector:
    def __init__(self):
        # MQTT Configuration
        self.mqtt_broker = "192.168.1.143"  # Default broker address
        self.mqtt_port = 1883  # Default broker port
        self.device_id = "openwrt-23"  # Default device ID
        self.metrics_topic = f"metrics/{self.device_id}"  # Metrics topic
        self.control_topic = f"control/{self.device_id}"  # Control topic for commands
        self.status_topic = f"status/{self.device_id}"  # Status topic
        self.anomalies_topic = f"anomalies/{self.device_id}"  # Anomalies topic
        
        # Operational Flags
        self.test_mode = True  # Default to test mode
        self.current_scenario = "normal"
        self.running = False
        self.collection_thread = None
        self.stop_event = threading.Event()
        self.is_trained = False 

        # Base Metrics for Test Mode
        self.base_metrics = {
            'client_count': 20,
            'channel_utilization': 0.4,
            'retry_rate': 0.1,
            'noise_floor': -85,
            'tx_failed_rate': 0.02
        }

        # MQTT Client Setup
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_disconnect = self.on_disconnect
        self.mqtt_client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback for MQTT connection"""
        if rc == 0:
            logging.info("Connected to MQTT broker")
            # Subscribe to relevant topics
            client.subscribe(self.control_topic)  # Subscribe to control topic
            client.subscribe(self.status_topic)  # Subscribe to status topic
            client.subscribe(self.anomalies_topic)  # Subscribe to anomalies topic
        else:
            logging.error(f"Failed to connect to MQTT broker, return code {rc}")

    def on_disconnect(self, client, userdata, rc):
        """Callback for MQTT disconnection"""
        logging.warning("Disconnected from MQTT broker")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            logging.info(f"Received MQTT message on topic {msg.topic}: {payload}")
            
            # Handle control commands
            if msg.topic == self.control_topic:
                command = payload.get("command")
                scenario = payload.get("scenario", "normal")
                test_mode = payload.get("test_mode", True)
                
                # Debug log: Processing control command
                logging.debug(f"Processing control command: {payload}")
                
                if command == "start":
                    logging.info(f"Changing scenario to: {scenario}")
                    self.current_scenario = scenario  # Update the scenario
                    
                    # Debug log: Updated current_scenario
                    logging.debug(f"Updated current_scenario to: {self.current_scenario}")
                    
                    self.start(test_mode=test_mode, scenario=scenario)
                elif command == "stop":
                    logging.info("Stopping scenario")
                    self.current_scenario = "normal"  # Reset to normal
                    
                    # Debug log: Updated current_scenario
                    logging.debug(f"Updated current_scenario to: {self.current_scenario}")
                    
                    self.stop()
        except Exception as e:
            logging.error(f"Error processing MQTT message: {str(e)}")

    def wait_for_training(self):
        """Wait for the model to be trained by checking the status topic"""
        while True:
            try:
                # Subscribe to status topic
                status_topic = f"status/{self.device_id}"
                status_received = None
                status_event = threading.Event()

                def on_status_message(client, userdata, msg):
                    nonlocal status_received
                    try:
                        payload = json.loads(msg.payload.decode('utf-8'))
                        status_received = payload
                        status_event.set()
                    except Exception as e:
                        logging.error(f"Error processing status message: {str(e)}")

                self.mqtt_client.subscribe(status_topic)
                self.mqtt_client.message_callback_add(status_topic, on_status_message)

                # Wait for status response
                if not status_event.wait(timeout=10):
                    logging.error("Timeout waiting for status response")
                    continue

                # Check if the model is trained
                if status_received.get('is_trained', False):
                    logging.info("Model is trained. Proceeding with tests.")
                    return
                else:
                    logging.info("Model not yet trained. Waiting...")
                    time.sleep(5)  # Wait before retrying
            except Exception as e:
                logging.error(f"Error waiting for training: {str(e)}")
                time.sleep(5)
            
    def check_training_status(self):
        """Check if the model is trained (non-blocking)"""
        try:
            status_topic = f"status/{self.device_id}"
            status_received = None

            def on_status_message(client, userdata, msg):
                nonlocal status_received
                try:
                    payload = json.loads(msg.payload.decode('utf-8'))
                    status_received = payload
                except Exception as e:
                    logging.error(f"Error processing status message: {str(e)}")

            # Subscribe to status topic temporarily
            self.mqtt_client.subscribe(status_topic)
            self.mqtt_client.message_callback_add(status_topic, on_status_message)
            
            # Give some time to receive the status
            time.sleep(2)
            
            if status_received and status_received.get('is_trained', False):
                self.is_trained = True
                logging.info("Model is trained. Proceeding with tests.")
        except Exception as e:
            logging.error(f"Error checking training status: {str(e)}")

    def collect_real_metrics(self) -> Dict:
        """Collect real metrics from OpenWRT system"""
        metrics = {}
        try:
            # Get client count and transmission stats
            station_cmd = ["iw", "dev", "phy0-ap0", "station", "dump"]
            station_output = subprocess.check_output(station_cmd, universal_newlines=True)
            
            # Process station data
            clients = []
            total_tx = 0
            total_retries = 0
            total_failed = 0
            current_station = None
            
            for line in station_output.split('\n'):
                if "Station" in line:
                    current_station = line.split()[1]
                    clients.append(current_station)
                elif current_station:
                    if "tx packets:" in line:
                        tx = int(line.split(':')[1].strip())
                        total_tx += tx
                    elif "tx retries:" in line:
                        retries = int(line.split(':')[1].strip())
                        total_retries += retries
                    elif "tx failed:" in line:
                        failed = int(line.split(':')[1].strip())
                        total_failed += failed
            
            metrics['client_count'] = len(clients)
            if total_tx > 0:
                metrics['retry_rate'] = total_retries / total_tx
                metrics['tx_failed_rate'] = total_failed / total_tx
            else:
                metrics['retry_rate'] = 0
                metrics['tx_failed_rate'] = 0
                
            # Get channel utilization and noise floor from survey
            survey_cmd = ["iw", "dev", "phy0-ap0", "survey", "dump"]
            survey_output = subprocess.check_output(survey_cmd, universal_newlines=True)
            
            channel_active = 0
            channel_busy = 0
            noise_floor = -95  # default
            
            for line in survey_output.split('\n'):
                if "channel active time:" in line:
                    channel_active = int(line.split(':')[1].strip().split()[0])
                elif "channel busy time:" in line:
                    channel_busy = int(line.split(':')[1].strip().split()[0])
                elif "noise:" in line:
                    noise_floor = int(line.split(':')[1].strip().split()[0])
            
            if channel_active > 0:
                metrics['channel_utilization'] = channel_busy / channel_active
            else:
                metrics['channel_utilization'] = 0
                
            metrics['noise_floor'] = noise_floor
            
            logging.info(f"Collected real metrics: {metrics}")
            return metrics
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Error collecting metrics: {e}")
            return self.generate_test_metrics()  # Fallback to test metrics

    def generate_test_metrics(self, scenario: str = "normal") -> Dict:
        """Generate test metrics based on scenario"""
        if scenario is None:
            scenario = self.current_scenario
        try:
            if scenario == "normal":
                noise_factor = 0.05
                metrics = {}
                for k, v in self.base_metrics.items():
                    if k == 'noise_floor':
                        metrics[k] = v + np.random.uniform(-2, 2)
                    else:
                        noise = np.random.uniform(-noise_factor, noise_factor) * abs(v)
                        metrics[k] = max(0, v + noise)
                return metrics
            
            elif scenario == "subtle":
                return {
                    'client_count': 23,
                    'channel_utilization': 0.6,
                    'retry_rate': 0.14, 
                    'noise_floor': -80,
                    'tx_failed_rate': 0.035
                }
            
            elif scenario == "obvious":
                return {
                    'client_count': 25,
                    'channel_utilization': 0.9,
                    'retry_rate': 0.25,
                    'noise_floor': -70,
                    'tx_failed_rate': 0.08
                }
            
            else:
                logging.warning(f"Unknown scenario: {scenario}, using normal")
                return self.generate_test_metrics("normal")
                
        except Exception as e:
            logging.error(f"Error generating test metrics: {e}")
            return self.base_metrics.copy()

    def send_metrics(self, metrics: Dict):
        """Send metrics to ML server via MQTT"""
        payload = {
            'timestamp': datetime.now().isoformat(),
            'device_id': self.device_id,
            'metrics': metrics
        }
        try:
            result, mid = self.mqtt_client.publish(self.metrics_topic, json.dumps(payload))
            if result != mqtt.MQTT_ERR_SUCCESS:
                logging.error(f"Failed to publish metrics: {mqtt.error_string(result)}")
            else:
                logging.info(f"Metrics sent via MQTT: {payload}")
        except Exception as e:
            logging.error(f"Error sending metrics via MQTT: {e}")

    def collection_loop(self):
        """Main collection loop"""
        while not self.stop_event.is_set():
            try:
                if self.test_mode:
                    # Generate test metrics based on scenario
                    metrics = self.generate_test_metrics(self.current_scenario)
                    logging.info(f"Generated test metrics for scenario {self.current_scenario}: {metrics}")
                else:
                    metrics = self.collect_real_metrics()
                    logging.info(f"Collected real metrics: {metrics}")
                self.send_metrics(metrics)
                time.sleep(5)  # 5-second intervals
            except Exception as e:
                logging.error(f"Error in collection loop: {e}")
                time.sleep(1)

    def start(self, test_mode: bool = True, scenario: str = "normal") -> bool:
        """Start the collector"""
        logging.info(f"Starting collector - test_mode: {test_mode}, scenario: {scenario}")
        
        if self.running:
            logging.warning("Collector already running")
            return False
            
        try:
            # Connect to MQTT broker
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.mqtt_client.loop_start()
            
            # if test_mode:
            #     self.wait_for_training()  # Wait for model to be trained
            
            self.test_mode = test_mode
            self.current_scenario = scenario
            self.stop_event.clear()
            
            self.collection_thread = threading.Thread(target=self.collection_loop)
            self.collection_thread.daemon = False
            self.collection_thread.start()
            
            self.running = True
            logging.info("Collector started successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error starting collector: {e}")
            self.running = False
            return False

    def stop(self) -> bool:
        """Stop the collector"""
        logging.info("Stopping collector")
        
        if not self.running:
            logging.warning("Collector not running")
            return False
            
        try:
            self.stop_event.set()
            if self.collection_thread:
                self.collection_thread.join(timeout=5)  # Wait for thread to finish
                if self.collection_thread.is_alive():
                    logging.error("Collection thread did not terminate within timeout")
            
            self.running = False
            logging.info("Collector stopped successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error stopping collector: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='OpenWRT Metrics Collector')
    
    # Add arguments with default values
    parser.add_argument('--broker', default="192.168.1.143", help='MQTT Broker Address')
    parser.add_argument('--port', type=int, default=1883, help='MQTT Broker Port')
    parser.add_argument('--topic', default="metrics/openwrt-23", help='MQTT Topic')
    parser.add_argument('--device-id', default="openwrt-23", help='Device ID')
    parser.add_argument('--api-key', default="test-key-12345", help='API Key')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode (default)')
    parser.add_argument('--real-mode', action='store_false', dest='test_mode', help='Run in real mode (collect real metrics)')
    
    args = parser.parse_args()
    
    # Initialize collector with default or provided arguments
    global collector
    collector = OpenWRTCollector()
    collector.mqtt_broker = args.broker
    collector.mqtt_port = args.port
    collector.metrics_topic = f"metrics/{args.device_id}"
    collector.control_topic = f"control/{args.device_id}"
    collector.status_topic = f"status/{args.device_id}"
    collector.anomalies_topic = f"anomalies/{args.device_id}"
    collector.device_id = args.device_id
    
    # Start the collector
    if collector.start(args.test_mode):
        logging.info("Collector started successfully")
    else:
        logging.error("Failed to start collector")

if __name__ == "__main__":
    main()
