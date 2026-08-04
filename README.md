# AI-Enabled Waste Segregation and Monitoring System

An intelligent waste management system that combines **Artificial Intelligence**, **Computer Vision**, and the **Internet of Things (IoT)** to automate waste segregation and optimize waste collection.

This project was developed as a research initiative and later published as a book chapter in *Advances in Electronics and Communication Systems: Design, Applications, and Emerging Technologies (Volume 6, 2026)*.

---

## Overview

Traditional waste management systems rely heavily on manual segregation and fixed garbage collection schedules. These approaches are inefficient, increase operational costs, reduce recycling efficiency, and expose sanitation workers to hazardous waste.

This project addresses these challenges by integrating **AI-powered waste classification** with **IoT-based bin monitoring** to create an automated and intelligent waste management solution.

---

## Problem Statement

Modern cities generate enormous amounts of municipal waste every day. Manual segregation is slow, expensive, and often inaccurate, while garbage collection vehicles frequently follow fixed routes regardless of whether bins are full or empty.

The objective of this project is to automate waste segregation and enable real-time monitoring of waste bins to improve recycling efficiency and optimize collection operations.

---

## Solution

The proposed system consists of two major modules:

### AI-Based Waste Segregation

A camera captures images of incoming waste.

A deep learning model running on the **NVIDIA Jetson Nano** classifies waste into:

* Dry Waste
* Wet Waste
* Plastic Waste
* Glass Waste

Based on the prediction, servo motors automatically direct the waste into the appropriate compartment.

### IoT-Based Waste Monitoring

An ultrasonic sensor continuously measures the fill level of the waste bin.

Once the bin reaches the configured threshold, the **SIM800C GSM module** automatically sends an alert to the responsible authority, ensuring waste is collected only when necessary.

---

## Features

* AI-powered real-time waste classification
* Edge AI inference using NVIDIA Jetson Nano
* Automatic waste segregation using servo motors
* Real-time waste level monitoring
* GSM-based SMS and call notifications
* Low-cost and scalable architecture
* Suitable for Smart City deployments

---

## System Architecture

```
Camera
   │
   ▼
Jetson Nano
   │
Deep Learning Model (PyTorch)
   │
Waste Classification
   │
Servo Motor Control
   │
Waste Segregation
──────────────────────────────────
Ultrasonic Sensor
   │
Arduino Uno
   │
SIM800C GSM Module
   │
SMS / Call Alert
```

---

## Technology Stack

### Artificial Intelligence

* PyTorch
* TorchVision
* OpenCV
* Convolutional Neural Networks (CNN)

### Hardware

* NVIDIA Jetson Nano
* Arduino Uno
* USB Camera
* HC-SR04 Ultrasonic Sensor
* SIM800C GSM Module
* Servo Motors

### Programming Languages

* Python
* C/C++ (Arduino)

---

## Experimental Results

* Classification Accuracy: **89.33%**
* Inference Time: **0.4–0.7 seconds**
* Ultrasonic Sensor Accuracy: **±1 cm**
* GSM Alert Delay: **4–6 seconds**

---

## Applications

* Smart Cities
* Hospitals
* Railway Stations
* Airports
* Shopping Malls
* Residential Communities
* Municipal Waste Management

---

## Future Improvements

* Cloud dashboard for centralized monitoring
* Solar-powered deployment
* Mobile application for waste collection teams
* GPS-based optimized collection routes
* Additional waste categories including hazardous materials

---

## Research Publication

This project has been published as a book chapter:

**Design and Implementation of an AI-Enabled Waste Segregation and Monitoring System**

Published in **Advances in Electronics and Communication Systems: Design, Applications, and Emerging Technologies (Volume 6, 2026).**

---

## Authors

* Priyansh Tiwari
* Anchal Mishra
* Ambuj Chaurasia
* Ashwini Parouha
* Pratik Sah

Guided by **Prof. Neeta Nathani**

---

## License

This repository is intended for academic and educational purposes.
