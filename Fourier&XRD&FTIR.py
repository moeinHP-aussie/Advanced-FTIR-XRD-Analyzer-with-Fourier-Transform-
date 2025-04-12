import subprocess
import sys

# لیست کتابخانه‌هایی که باید نصب شوند
required_libraries = [
    'numpy',
    'scipy',
    'pandas',
    'scikit-learn',
    'matplotlib',
    'tkinter',  
]

# نصب کتابخانه‌ها اگر موجود نباشند
for library in required_libraries:
    try:
        __import__(library)
    except ImportError:
        print(f"{library} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", library])

#####################################################

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from scipy.fft import fft, ifft
import pandas as pd
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class FourierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced FTIR/XRD Analyzer")
        self.root.geometry("1200x900")
        
        # متغیرهای حالت
        self.sample_rate = 1000
        self.real_data = None
        self.wavenumbers = None
        self.last_fft = None
        
        # ایجاد اجزای رابط کاربری
        self.create_widgets()
        self.create_plots()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=(15, 10))
        main_frame.pack(fill=tk.BOTH, expand=True)

        # پنل کنترل
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=(10, 5))
        control_frame.grid(row=0, column=0, padx=10, pady=5, sticky=tk.NSEW)

        # ورودی تابع سیگنال
        ttk.Label(control_frame, text="Signal Function:").grid(row=0, column=0, sticky=tk.W)
        self.func_entry = ttk.Entry(control_frame, width=40)
        self.func_entry.grid(row=0, column=1, columnspan=2, sticky=tk.EW)
        self.func_entry.insert(0, "np.sin(2*np.pi*5*t) + 0.3*np.cos(2*np.pi*20*t)")

        # دکمه آپلود داده
        ttk.Button(control_frame, text="Upload Data", command=self.load_file).grid(row=1, column=0, pady=5)
        
        # انتخاب نوع عملیات
        self.operation_var = tk.StringVar()
        operations = [
            "Generate Signal",
            "Fourier Transform",
            "Inverse Fourier Transform",
            "Fourier Series",
            "FTIR/XRD Advanced Analysis"
        ]
        self.operation_menu = ttk.Combobox(control_frame, textvariable=self.operation_var, values=operations)
        self.operation_menu.grid(row=1, column=1, padx=5, sticky=tk.EW)
        self.operation_menu.current(0)
        
        # انتخاب نوع تحلیل
        self.analysis_type_var = tk.StringVar()
        self.analysis_type_menu = ttk.Combobox(
            control_frame, 
            textvariable=self.analysis_type_var, 
            values=["FTIR", "XRD"], 
            state="readonly"
        )
        self.analysis_type_menu.grid(row=1, column=2, padx=5, sticky=tk.EW)
        self.analysis_type_menu.current(0)

        # دکمه اجرا
        ttk.Button(control_frame, text="Run", command=self.run_operation).grid(row=1, column=3)

        # ناحیه نمودار
        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=1, column=0, padx=10, pady=5, sticky=tk.NSEW)

        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

    def create_plots(self):
        self.ax.clear()
        self.ax.grid(True)
        self.canvas.draw()

    def load_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx"), ("Text Files", "*.txt")]
        )
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path, skiprows=1, header=0, delimiter=',')
                elif file_path.endswith('.xlsx'):
                    # خواندن شیت اول (میتوانید نام یا شماره شیت را تغییر دهید)
                    df = pd.read_excel(file_path, sheet_name=0)
                    # تغییر نام ستونها به Angle و Intensity (مطابق فایل شما)
                    df.columns = ['Angle', 'Intensity']
                else:
                    raise ValueError("Unsupported file format")
                
                # ذخیره دادهها
                self.wavenumbers = df.iloc[:, 0].values
                self.real_data = df.iloc[:, 1].values
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read file: {str(e)}")

    def generate_signal(self):
        t = np.linspace(-1, 4, self.sample_rate)
        try:
            # دیکشنری پیشرفته‌تر برای توابع و ویژگی‌های numpy
            custom_globals = {
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan,
                'exp': np.exp, 'log': np.log, 'log10': np.log10, 'sqrt': np.sqrt, 'abs': np.abs, 'ceil': np.ceil,
                'floor': np.floor, 'round': np.round, 'pi': np.pi, 'e': np.e, 'inf': np.inf, 'nan': np.nan,
                'sinusoidal': lambda t, A, f, phi: A * np.sin(2 * np.pi * f * t + phi),  # یک مثال از تابع سفارشی
                't': t, 'np': np
            }
            # ارزیابی ورودی کاربر با استفاده از دیکشنری سفارشی
            signal = eval(self.func_entry.get(), custom_globals)
            equation = f"Function: {self.func_entry.get()}\nSampling Rate: {self.sample_rate} Hz"
            return t, signal, equation
        except Exception as e:
            messagebox.showerror("Error", f"Invalid function: {e}")
            return None, None, None


    def compute_fft(self, signal):
        N = len(signal)
        fft_result = fft(signal)
        freq = np.fft.fftfreq(N, 1/self.sample_rate)
        self.last_fft = fft_result
        
        magnitudes = np.abs(fft_result)
        peak_freqs = freq[np.where(magnitudes > 0.5*np.max(magnitudes))]
        equation = "Fourier Transform:\nX[k] = Σ x[n]*e^(-j2πkn/N)\n"
        equation += f"Peak Frequencies: {', '.join([f'{f:.1f} Hz' for f in peak_freqs if f > 0])}"
        return freq[:N//2], magnitudes[:N//2], equation

    def compute_ifft(self):
        if self.last_fft is None:
            raise ValueError("Perform FFT first!")
        reconstructed = ifft(self.last_fft).real
        equation = "Inverse Fourier Transform:\nx[n] = (1/N)Σ X[k]*e^(j2πkn/N)"
        return np.linspace(-1, 4, len(reconstructed)), reconstructed, equation

    def fourier_series(self, t, num_terms=5):
        signal = np.zeros_like(t)
        series_eq = f"Fourier Series ({num_terms} terms):\nf(t) = Σ (1/n)*sin(2πnt)\nn = 1 to {num_terms}"
        for n in range(1, num_terms+1):
            signal += (1/n) * np.sin(2 * np.pi * n * t)
        return t, signal, series_eq

    def analyze_ftir(self):
        if self.real_data is None or self.wavenumbers is None:
            messagebox.showwarning("Warning", "Upload FTIR data first!")
            return None
        
        data = 100 - self.real_data  # تبدیل به جذب
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        peaks_indices, _ = find_peaks(data_norm, height=0.3, prominence=0.1, distance=10)
        
        if len(peaks_indices) == 0:
            messagebox.showwarning("Warning", "No significant peaks found!")
            return None
        
        features = self.wavenumbers[peaks_indices].reshape(-1, 1)
        n_clusters = min(3, len(peaks_indices))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
        
        eq = "FTIR Analysis:\n"
        eq += f"Detected {len(peaks_indices)} major absorption bands\n"
        eq += "Peak positions (cm⁻¹):\n" + ", ".join([f"{self.wavenumbers[i]:.1f}" for i in peaks_indices])
        
        return self.wavenumbers, data_norm, peaks_indices, kmeans.labels_, eq

    def analyze_xrd(self):
        if self.real_data is None or self.wavenumbers is None:
            messagebox.showwarning("Warning", "Upload XRD data first!")
            return None
        
        data = (self.real_data - np.min(self.real_data)) / (np.max(self.real_data) - np.min(self.real_data))
        smooth_data = np.convolve(data, np.ones(10)/10, mode='same')
        
        peaks_indices, _ = find_peaks(smooth_data, prominence=0.1, distance=10)
        
        if len(peaks_indices) == 0:
            messagebox.showwarning("Warning", "No significant peaks found!")
            return None
        
        features = self.wavenumbers[peaks_indices].reshape(-1, 1)
        n_clusters = min(3, len(peaks_indices))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
        
        eq = "XRD Analysis:\n"
        eq += f"Detected {len(peaks_indices)} diffraction peaks\n"
        eq += "Peak positions (2θ):\n" + ", ".join([f"{self.wavenumbers[i]:.1f}" for i in peaks_indices])
        
        return self.wavenumbers, smooth_data, peaks_indices, kmeans.labels_, eq

    def update_plot(self, x, y, equation, peaks=None, labels=None, additional_info=""):
        self.ax.clear()
        self.ax.plot(x, y, linewidth=1, color='b')
        
        if peaks is not None:
            self.ax.scatter(x[peaks], y[peaks], c=labels, marker='x', s=50, 
                           cmap='viridis', zorder=2)
            
        self.ax.set_xlabel('Wavenumber (cm⁻¹)' if 'FTIR' in equation else '2θ Angle')
        self.ax.set_ylabel('Absorbance (Normalized)' if 'FTIR' in equation else 'Intensity (Normalized)')
        self.ax.grid(True, alpha=0.3)
        
        info_text = equation + "\n" + additional_info
        self.ax.text(0.98, 0.98, info_text, transform=self.ax.transAxes,
                    fontsize=8, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.8))
        
        self.canvas.draw()

    def run_operation(self):
        operation = self.operation_var.get()
        
        try:
            if operation == "Generate Signal":
                t, signal, eq = self.generate_signal()
                if signal is not None:
                    self.update_plot(t, signal, eq)

            elif operation == "Fourier Transform":
                t, signal, _ = self.generate_signal()
                if signal is None: 
                    return
                freq, mag, eq = self.compute_fft(signal)
                self.update_plot(freq, mag, eq)

            elif operation == "Inverse Fourier Transform":
                t, signal, eq = self.compute_ifft()
                self.update_plot(t, signal, eq)

            elif operation == "Fourier Series":
                t, signal, eq = self.fourier_series(np.linspace(-1, 4, 1000))
                self.update_plot(t, signal, eq)

            elif operation == "FTIR/XRD Advanced Analysis":
                analysis_type = self.analysis_type_var.get()
                
                if analysis_type == "FTIR":
                    result = self.analyze_ftir()
                    if result is None: 
                        return
                    x, y, peaks, labels, eq = result
                    additional_info = f"Cluster centers: {np.unique([f'{x[i]:.1f} cm⁻¹' for i in peaks])}"
                
                elif analysis_type == "XRD":
                    result = self.analyze_xrd()
                    if result is None: 
                        return
                    x, y, peaks, labels, eq = result
                    additional_info = f"Cluster centers: {np.unique([f'{x[i]:.1f}°' for i in peaks])}"
                
                self.update_plot(x, y, eq, peaks, labels, additional_info)

        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = FourierApp(root)
    root.mainloop()