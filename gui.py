import customtkinter as ctk
import pandas as pd
import numpy as np
import onnxruntime as ort
import joblib
import time
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "smart_grid_forecast.onnx")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "CurrentVoltage.csv")
WINDOW_SIZE = 60

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class SmartGridApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AMD RYZEN AI - Smart Grid Transfomer")
        self.geometry("900x550")
        self.session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        self.scaler = joblib.load(SCALER_PATH)
        self.df = pd.read_csv(DATA_PATH)
        self.current_row = self.df[self.df['IL1'] > 0].index[0]
        self.setup_ui()
        self.update_loop()

    def setup_ui(self):
        self.grid_columnconfigure((0, 1), weight=1)

        # Header
        self.logo_label = ctk.CTkLabel(self, text="RYZEN AI: Smart Grid Transformer", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, columnspan=2, padx=20, pady=20)

        # Voltage Card
        self.v_frame = ctk.CTkFrame(self)
        self.v_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        ctk.CTkLabel(self.v_frame, text="VOLTAGE (V)", text_color="#00E5FF").pack(pady=10)
        self.v_label = ctk.CTkLabel(self.v_frame, text="0.0", font=ctk.CTkFont(size=48, family="Consolas"))
        self.v_label.pack(pady=10)

        # Current Card
        self.i_frame = ctk.CTkFrame(self)
        self.i_frame.grid(row=1, column=1, padx=20, pady=10, sticky="nsew")
        ctk.CTkLabel(self.i_frame, text="CURRENT (A)", text_color="#FFEA00").pack(pady=10)
        self.i_label = ctk.CTkLabel(self.i_frame, text="0.0", font=ctk.CTkFont(size=48, family="Consolas"))
        self.i_label.pack(pady=10)

        # Risk Bars Frame
        self.risk_frame = ctk.CTkFrame(self)
        self.risk_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=20, sticky="nsew")

        # Live Risk
        ctk.CTkLabel(self.risk_frame, text="LIVE OVERLOAD RISK").pack()
        self.live_bar = ctk.CTkProgressBar(self.risk_frame, width=700)
        self.live_bar.pack(pady=10)
        self.live_bar.set(0)

        # Forecast Risk
        ctk.CTkLabel(self.risk_frame, text="15-MINUTE FORECASTED RISK", text_color="#A020F0").pack()
        self.forecast_bar = ctk.CTkProgressBar(self.risk_frame, width=700, progress_color="#A020F0")
        self.forecast_bar.pack(pady=10)
        self.forecast_bar.set(0)

        # Status and Latency
        self.status_btn = ctk.CTkButton(self, text="SYSTEM STABLE", fg_color="green", state="disabled")
        self.status_btn.grid(row=3, column=0, columnspan=2, pady=10)

        self.latency_label = ctk.CTkLabel(self, text="Inference: -- ms", font=ctk.CTkFont(size=12))
        self.latency_label.grid(row=4, column=0, columnspan=2)

    def update_loop(self):
        if self.current_row >= len(self.df): self.current_row = 0

        start_idx = max(0, self.current_row - WINDOW_SIZE)
        window_data = self.df.iloc[start_idx:self.current_row][['VL1', 'IL1']].values
        if len(window_data) < WINDOW_SIZE:
            window_data = np.vstack([np.zeros((WINDOW_SIZE - len(window_data), 2)), window_data])

        scaled_window = self.scaler.transform(window_data).astype(np.float32)
        input_data = np.expand_dims(scaled_window, axis=0)
        
        start_t = time.perf_counter()
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_data})
        latency = (time.perf_counter() - start_t) * 1000
        
        prob_now, prob_future = outputs[0][0][0], outputs[0][0][1]

        self.v_label.configure(text=f"{self.df.iloc[self.current_row]['VL1']:.1f}")
        self.i_label.configure(text=f"{self.df.iloc[self.current_row]['IL1']:.1f}")
        self.live_bar.set(prob_now)
        self.forecast_bar.set(prob_future)
        self.latency_label.configure(text=f"Ryzen AI Engine Latency: {latency:.2f}ms")
 
        if prob_now > 0.5:
            self.status_btn.configure(text="CRITICAL: OVERLOAD", fg_color="red")
        elif prob_future > 0.7:
            self.status_btn.configure(text="WARNING: FUTURE TRIP DETECTED", fg_color="orange")
        else:
            self.status_btn.configure(text="SYSTEM STABLE", fg_color="green")

        self.current_row += 1
        self.after(1000, self.update_loop)

if __name__ == "__main__":
    app = SmartGridApp()
    app.mainloop()