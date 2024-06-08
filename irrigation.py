#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:03:00 2024

@author: ishikaishani
"""

import numpy as np
import pickle
import streamlit as st

# Load the Q-table model
q_table = pickle.load(open('/Users/ishikaishani/Desktop/techbrust/qtable_model.sav', 'rb'))

def irrigation_prediction(input_data, q_table):
    current_state_continuous = np.asarray(input_data, dtype=np.float32)
    n_bins = 10
    bins = [np.linspace(0, 1, n_bins) for _ in range(len(current_state_continuous))]

    def discretize_state(state):
        """Convert a continuous state to a discrete state."""
        discretized_state = []
        for i, bin_ in enumerate(bins):
            discretized_state.append(np.digitize(state[i], bin_) - 1)  # -1 to get zero-based index
        return tuple(discretized_state)
    
    # Discretize the current state
    current_state_discrete = discretize_state(current_state_continuous)
    
    # Use the Q-table to decide on the action
    action = np.argmax(q_table[current_state_discrete])
    
    # Output the action
    if action == 1:
        return 'Irrigate'
    else:
        return 'Do not irrigate'
    
def main():
    st.title('Irrigation Prediction')
    CropDays = st.text_input('No of Crop Days:')
    Temperature = st.text_input('Temperature:')
    Humidity = st.text_input('Humidity:')
    SoilMoisture = st.text_input('Soil Moisture:')
    
    # Code for prediction
    diagnosis = ''
    
    # Creating a button for prediction
    if st.button('Irrigation_Need'):
        try:
            input_data = [
                float(CropDays),
                float(Temperature),
                float(Humidity),
                float(SoilMoisture)
            ]
            diagnosis = irrigation_prediction(input_data, q_table)
        except ValueError:
            diagnosis = "Invalid input. Please enter numeric values."
        
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()
