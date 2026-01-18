from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

model, le = joblib.load("lamp_ai_full.pkl")

