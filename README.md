# GPX Mining Project

## Overview

This project analyzes airborne geophysical survey data from the Kobold Metals Sitatunga survey in Zambia. Processed gravity, magnetic, and radiometric data to identify potential mineral deposits and optimize exploration efforts.

## Goal

The main goal is to help mining companies save money by targeting specific areas for field testing instead of conducting broad surveys. By using machine learning to detect anomalies and patterns in geophysical data, we can send geologists to the most promising locations, reducing travel costs and increasing discovery success rates.

## Data

The survey covered 23,829 line-kilometres in Zambia between August-October 2021, collecting three types of geophysical data:

- **Gravity data** (20 features): Measures variations in Earth's gravitational field
- **Magnetic data** (21 features): Measures variations in Earth's magnetic field  
- **Radiometric data** (22 features): Measures gamma ray decay of surface materials

## Installation

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
``` 