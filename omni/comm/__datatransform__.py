from .__omni__ import *
import matplotlib.pyplot as plt
from io import BytesIO
import pywt
from PyEMD import EMD,CEEMDAN
from vmdpy import VMD
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew
from scipy.signal import ShortTimeFFT



# Time-frequency characteristic 时频特征
# ==========================================================================================================

def Time_Frequency_characteristic(data: array) -> tuple:
    # 统计量

    mean_complex = np.mean(abs(data))       # 均值
    var_complex = np.var(data)         # 方差
    power = np.mean(np.abs(data)**2)   # 功率
    kurt_real = kurtosis(data.real, fisher=True)  # 实部峰度 如果不用fisher＞3说明分布尖锐，＜3平坦   用了fisher就相当于-3  ＞0尖锐，＜0平坦
    kurt_imag = kurtosis(data.imag, fisher=True)  # 虚部峰度
    skew_real = skew(data.real)  # 实部偏度 衡量对称性，＞0右偏，＜0左偏
    skew_imag = skew(data.imag)  # 虚部偏度

    # 高阶矩
    M2 = np.mean(data ** 2)  # 二阶矩
    M3 = np.mean(data ** 3)  # 三阶矩
    M4 = np.mean(data ** 4)  # 四阶矩

    # 高阶累积量 前三阶和高阶矩是一样的 第4阶开始不一样
    C4 = M4 - 2 * (M2 ** 2) - np.abs(M2) ** 2  # 四阶累积量
    return mean_complex,var_complex,power,kurt_real,kurt_imag,skew_real,skew_imag,M2,M3,M4,C4

# bispectrum 双谱
# ==========================================================================================================
def compute_bispectrum(signal, n_fft=256):

    N = len(signal)
    if N < n_fft:
        raise ValueError("信号长度必须大于或等于n_fft")

    segment_count = N // n_fft
    bispectrum = np.zeros((n_fft, n_fft), dtype=complex)

    for i in range(segment_count):
        segment = signal[i * n_fft:(i + 1) * n_fft]
        fft_segment = np.fft.fft(segment, n=n_fft)

        for f1 in range(n_fft):
            for f2 in range(n_fft):
                f3 = (f1 + f2) % n_fft  # 第三个频率分量
                bispectrum[f1, f2] += (
                        fft_segment[f1] * fft_segment[f2] * np.conj(fft_segment[f3])
                )
    # 归一化双谱
    bispectrum /= segment_count
    freqs = np.fft.fftfreq(n_fft)
    return bispectrum, freqs

def bispectrum(data):
    # 计算双谱
    bispectrum, freqs = compute_bispectrum(data, n_fft=256)

    # 取双谱的模值
    bispectrum_magnitude = np.abs(bispectrum)
    bispectrum_log = np.log(bispectrum_magnitude + 1),


    # 绘制双谱的模值
    plt.figure(figsize=(8, 6))
    plt.imshow(
        np.log(bispectrum_magnitude + 1),  # 对数压缩以突出细节
        extent=(freqs[0], freqs[-1], freqs[0], freqs[-1]),
        origin="lower",
        cmap="jet",
        aspect="auto",
    )
    plt.colorbar(label="Magnitude (Log Scale)")
    plt.title("Bispectrum Magnitude")
    plt.xlabel("Frequency 1")
    plt.ylabel("Frequency 2")
    plt.show()

    return bispectrum
    #return bispectrum_log #如果return这里的话就是对双谱进行一些数据放缩


# bispectrum 三谱
# ==========================================================================================================

def compute_trispectrum(signal, n_fft=256):
    N = len(signal)
    if N < n_fft:
        raise ValueError("信号长度必须大于或等于n_fft")

    # 分段处理
    segment_count = N // n_fft
    trispectrum = np.zeros((n_fft, n_fft, n_fft), dtype=complex)

    for i in range(segment_count):
        segment = signal[i * n_fft:(i + 1) * n_fft]
        fft_segment = np.fft.fft(segment, n=n_fft)

        for f1 in range(n_fft):
            print(1)
            for f2 in range(n_fft):
                for f3 in range(n_fft):
                    f4 = (-f1 - f2 - f3) % n_fft  # 第四个频率分量
                    trispectrum[f1, f2, f3] += (
                            fft_segment[f1] * fft_segment[f2] *
                            np.conj(fft_segment[f3]) * np.conj(fft_segment[f4])
                    )
    # 归一化三谱
    trispectrum /= segment_count
    freqs = np.fft.fftfreq(n_fft)
    return trispectrum, freqs

def trispectrum(data):
    # 计算双谱
    trispectrum, freqs = compute_trispectrum(data, n_fft=512)

    # 取双谱的模值
    trispectrum_magnitude = np.abs(trispectrum)
    trispectrum_log = np.log(trispectrum_magnitude + 1),
    # # 绘制双谱的模值
    plt.figure(figsize=(8, 6))
    plt.imshow(
        np.log(trispectrum_magnitude + 1),  # 对数压缩以突出细节
        extent=(freqs[0], freqs[-1], freqs[0], freqs[-1]),
        origin="lower",
        cmap="jet",
        aspect="auto",
    )
    plt.colorbar(label="Magnitude (Log Scale)")
    plt.title("Bispectrum Magnitude")
    plt.xlabel("Frequency 1")
    plt.ylabel("Frequency 2")
    plt.show()
    return trispectrum
    #return bispectrum_log


# Constellation 星座图
# ==========================================================================================================
def Constellation(data: array) -> array:
    # 创建图像数据对象设置参数
    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    ax.scatter(data.real, data.imag, s=1, color='blue')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.axis('off')
    # 获取图像数据以用于后续储存为数据
    buf = BytesIO()
    fig.savefig(buf, format='png')  # 可以选择其他格式，如'pdf', 'svg', 但对于像素数据通常使用'png'或'rgba'
    buf.seek(0)
    image_data = plt.imread(buf)  # 读取为NumPy数组
    plt.show()
    #print(image_data.shape)

    # 如果需要RGB格式而不是RGBA,A代表透明度
    # if image_data.shape[2] == 4:
    #     image_data_rgb = image_data[:, :, :3]  # 去掉alpha通道
    # else:
    #     image_data_rgb = image_data  # 已经是RGB或灰度图
    return image_data



# Moderate trajectory diagram 矢量图
# ==========================================================================================================
def Trajectory(data: array) -> array:
    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)  # 224x224分辨率
    ax.plot(data.real, data.imag, linestyle='-', linewidth=0.5, color='blue')  # 连线较细

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')  # 关闭轴
    ax.set_aspect('equal')  # 保持坐标比例一致

    buf = BytesIO()
    fig.savefig(buf, format='png')  # 可以选择其他格式，如'pdf', 'svg', 但对于像素数据通常使用'png'或'rgba'
    buf.seek(0)
    image_data = plt.imread(buf)  # 读取为NumPy数组
    plt.show()

    return image_data



# Vector image and constellation image fusion 矢量星座图
# ==========================================================================================================


def fusion(data: array) -> array:
    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)  # 224x224分辨率
    ax.plot(data.real, data.imag, linestyle='-', marker='o', markersize=5, linewidth=0.5, color='blue',markerfacecolor='red')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')  # 关闭轴
    ax.set_aspect('equal')  # 保持坐标比例一致

    buf = BytesIO()
    fig.savefig(buf, format='png')  # 可以选择其他格式，如'pdf', 'svg', 但对于像素数据通常使用'png'或'rgba'
    buf.seek(0)
    image_data = plt.imread(buf)  # 读取为NumPy数组
    plt.show()

    return image_data



# Wavelet transform 小波变换
# ==========================================================================================================

def Wavelet_transform(data):

    wavelet = 'cmor1.5-1.0'   # 时间宽度参数-频率宽度参数
    #wavelet = 'shan1.5-1.0'
    #wavelet = 'mexh'
    #wavelet = 'gaus3'

    coeffs_real, freqs = pywt.cwt(data.real, scales=np.arange(1, 128), wavelet=wavelet)
    coeffs_imag, freqs = pywt.cwt(data.imag, scales=np.arange(1, 128), wavelet=wavelet)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.imshow(np.abs(coeffs_real), extent=[0, len(data), freqs[-1], freqs[0]], aspect='auto', cmap='jet')
    plt.title('Real Part - Magnitude of Continuous Wavelet Transform')
    plt.colorbar()
    plt.subplot(2, 1, 2)
    plt.imshow(np.abs(coeffs_imag), extent=[0, len(data), freqs[-1], freqs[0]], aspect='auto', cmap='jet')
    plt.title('Imaginary Part - Magnitude of Continuous Wavelet Transform')
    plt.colorbar()
    plt.tight_layout()
    plt.show()



# 眼图
# ==========================================================================================================

def plot_eye_diagram(signal, samples_per_symbol):
    num_segments = len(signal) // samples_per_symbol
    segments = signal[:num_segments * samples_per_symbol].reshape(num_segments, samples_per_symbol)

    plt.figure(figsize=(10, 6))
    for segment in segments:
        plt.plot(np.arange(samples_per_symbol), segment.real, color='blue', alpha=0.2)  # 实部
        plt.plot(np.arange(samples_per_symbol), segment.imag, color='red', alpha=0.2)  # 虚部
    plt.title('Eye Diagram of QPSK Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()


    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)  # 224x224分辨率
    ax.plot(np.arange(samples_per_symbol), segment.real, color='blue', alpha=0.2)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')  # 关闭轴
    ax.set_aspect('equal')  # 保持坐标比例一致

    buf = BytesIO()
    fig.savefig(buf, format='png')  # 可以选择其他格式，如'pdf', 'svg', 但对于像素数据通常使用'png'或'rgba'
    buf.seek(0)
    image_data = plt.imread(buf)  # 读取为NumPy数组
    plt.show()

    return image_data




from scipy.signal import stft
# Vector image and constellation image fusion STFT
# ==========================================================================================================
from scipy.signal.windows import gaussian
def STFT(data):

    nperseg = 64  # 每段窗口长度
    g_std = 8
    w = gaussian(50, std=g_std, sym=True)  # symmetric Gaussian window
    stft1 = ShortTimeFFT(win=w,fs=1000,hop=nperseg,fft_mode='centered')
    Zxx = stft1.stft(data)
    X, Y = np.arange(Zxx.shape[1]), np.arange(Zxx.shape[0])
    X, Y = np.meshgrid(X, Y)

    # # 绘制 STFT 的时频图
    plt.figure(figsize=(10, 6))

    plt.pcolormesh(X,Y, 20 * np.log10(np.abs(Zxx) + 1e-8), shading='gouraud')

    plt.colorbar(label='Magnitude (dB)')
    plt.title('STFT Magnitude Spectrum')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.show()

    return Zxx

#
# def STFT(data, fs=1000, nperseg=64, hop=None, g_std=8):
#     if hop is None:
#         hop = nperseg  # 如果没有指定重叠，则使用窗口大小作为重叠
#
#     # 生成对称高斯窗
#     w = gaussian(nperseg, std=g_std, sym=True)
#
#     # 计算 STFT
#     f, t, Zxx = stft(data, fs=fs, nperseg=nperseg, noverlap=nperseg - hop, window=w, boundary='zeros',return_onesided=False)
#
#     # 转换到 dB 尺度
#     magnitude_db = 20 * np.log10(np.abs(Zxx) + 1e-8)
#
#     # 绘制 STFT 的时频图
#     plt.figure(figsize=(10, 6))
#     plt.pcolormesh(t, f, magnitude_db, shading='gouraud', cmap='viridis')
#     plt.colorbar(label='Magnitude (dB)')
#     plt.title('STFT Magnitude Spectrum')
#     plt.ylabel('Frequency (Hz)')
#     plt.xlabel('Time (s)')
#     plt.tight_layout()  # 调整布局以避免标签重叠
#     plt.show()
#
#     return Zxx

# Vector image and constellation image fusion 经验模态分解EMD
# ==========================================================================================================

def EMD_EMD(data: array) -> tuple:

    emd_real = EMD()
    emd_imag = EMD()
    num_of_IMFs = 10 # 设置模态数
    IMFs_real = emd_real.emd(data.real,max_imf=num_of_IMFs)
    IMFs_imag = emd_imag.emd(data.imag,max_imf=num_of_IMFs)

    # 实部IMF结果
    N = IMFs_real.shape[0] + 1  # 加上原始信号
    t = np.arange(len(data.real))

    plt.figure(figsize=(20, 16))
    plt.subplot(N, 2, 1)  # N行，2列，第1个，顺序为从左到右，从上到下
    plt.plot(t, data.real, 'r')
    plt.title("Real Original Signal")
    for i in range(IMFs_real.shape[0]):
        plt.subplot(N, 2, 2 * i + 3)
        plt.plot(t, IMFs_real[i], 'g')
        plt.title(f"Real_IMF {i + 1}")
    plt.subplot(N, 2, 2 * N - 1)
    plt.plot(t, emd_real.residue, 'b')
    plt.title("Real_Residue")

    # 虚部IMF结果
    plt.subplot(N, 2, 2)
    plt.plot(t, data.imag, 'r')
    plt.title("Imag Original Signal")
    for i in range(IMFs_imag.shape[0]):
        plt.subplot(N, 2, 2 * i + 3 + 1)
        plt.plot(t, IMFs_imag[i], 'g')
        plt.title(f"Imag_IMF {i + 1}")
    plt.subplot(N, 2, 2 * N)
    plt.plot(t, emd_imag.residue, 'b')
    plt.title("Imag_Residue")
    plt.tight_layout()
    plt.show()

    return IMFs_real, IMFs_imag

def EMD_CEEMDAN(data: array) -> tuple:
    #CEEMDAN里的残余项
    emd_real = CEEMDAN()
    emd_imag = CEEMDAN()
    num_of_IMFs = 10 # 设置模态数
    IMFs_real = emd_real.emd(data.real,max_imf=num_of_IMFs)
    IMFs_imag = emd_imag.emd(data.imag,max_imf=num_of_IMFs)

    # 实部IMF结果
    N = IMFs_real.shape[0] + 1  # 加上原始信号
    t = np.arange(len(data.real))

    plt.figure(figsize=(20, 16))
    plt.subplot(N, 2, 1)  # N行，2列，第1个，顺序为从左到右，从上到下
    plt.plot(t, data.real, 'r')
    plt.title("Real Original Signal")
    for i in range(IMFs_real.shape[0]):
        plt.subplot(N, 2, 2 * i + 3)
        plt.plot(t, IMFs_real[i], 'g')
        plt.title(f"Real_IMF {i + 1}")
    plt.subplot(N, 2, 2 * N - 1)
    plt.title("Real_Residue")

    # 虚部IMF结果
    plt.subplot(N, 2, 2)
    plt.plot(t, data.imag, 'r')
    plt.title("Imag Original Signal")
    for i in range(IMFs_imag.shape[0]):
        plt.subplot(N, 2, 2 * i + 3 + 1)
        plt.plot(t, IMFs_imag[i], 'g')
        plt.title(f"Imag_IMF {i + 1}")
    plt.subplot(N, 2, 2 * N)
    plt.title("Imag_Residue")
    plt.tight_layout()
    plt.show()
    return IMFs_real, IMFs_imag




# Vector image and constellation image fusion 希尔伯特-黄
# ==========================================================================================================

def hilbert_transform(imfs):

    #hilbert_result = []
    for imf in imfs:
        analytic_signal = hilbert(imf)
        #amplitude = np.abs(analytic_signal)  # 瞬时幅值
        #phase = np.angle(analytic_signal)   # 瞬时相位
        #hilbert_result.append((amplitude, phase))
    return analytic_signal
    #return hilbert_result

def hilbert_Huang(data: array) -> tuple:
    #emd_real,emd_imag = EMD_EMD(data)
    emd_real,emd_imag = EMD_CEEMDAN(data)
    hilbert_real = hilbert_transform(emd_real)
    hilbert_imag = hilbert_transform(emd_imag)

    return hilbert_real,hilbert_imag


# Vector image and constellation image fusion 变分模态分解
# ==========================================================================================================

def VMD_VMD(data: array) -> tuple:

    # alpha 惩罚系数；带宽限制经验取值为抽样点长度1.5-2.0倍.
    # 惩罚系数越小，各IMF分量的带宽越大，过大的带宽会使得某些分量包含其他分量言号;
    # a值越大，各IMF分量的带宽越小，过小的带宽是使得被分解的信号中某些信号丢失该系数常见取值范围为1000~3000
    alpha = 2000
    tau = 0         # tau 噪声容限，即允许重构后的信号与原始信号有差别。
    K = 10           # K 分解模态（IMF）个数
    DC = 0          # DC 若为0则让第一个IMF为直流分量/趋势向量
    init = 1        # init 指每个IMF的中心频率进行初始化。当初始化为1时，进行均匀初始化。
    tol = 1e-7      # 控制误差大小常量，决定精度与迭代次数

    u_real, u_hat_real, omega_real = VMD(data.real, alpha, tau, K, DC, init, tol)  # 输出U是各个IMF分量，u_hat是各IMF的频谱，omega为各IMF的中心频率
    u_imag, u_hat_imag, omega_imag = VMD(data.real, alpha, tau, K, DC, init, tol)  # 输出U是各个IMF分量，u_hat是各IMF的频谱，omega为各IMF的中心频率

    # 绘制时域结果
    # 实部IMF结果
    N = u_real.shape[0] + 1  # 加上原始信号
    t = np.arange(len(data.real))

    plt.figure(figsize=(20, 16))
    plt.subplot(N, 2, 1)  # N行，2列，第1个，顺序为从左到右，从上到下
    plt.plot(t, data.real, 'r')
    plt.title("Real Original Signal")

    for i in range(u_real.shape[0]):
        plt.subplot(N, 2, 2 * i + 3)
        plt.plot(t, u_real[i], 'g')
        plt.title(f"Real_IMF {i + 1}")
    plt.subplot(N, 2, 2 * N - 1)
    plt.title("Real_Residue")

    # 虚部IMF结果
    plt.subplot(N, 2, 2)
    plt.plot(t, data.imag, 'r')
    plt.title("Imag Original Signal")
    for i in range(u_imag.shape[0]):
        plt.subplot(N, 2, 2 * i + 3 + 1)
        plt.plot(t, u_imag[i], 'g')
        plt.title(f"Imag_IMF {i + 1}")
    plt.subplot(N, 2, 2 * N)
    plt.title("Imag_Residue")

    plt.tight_layout()
    plt.show()

    # 绘制频域结果
    plt.figure(figsize=(20, 16))

    # 实部频谱
    for i in range(K):
        plt.subplot(K, 2, 2*i+1 )
        plt.plot(np.abs(u_hat_real[:, i]))
        plt.title('（VMD）amplitude frequency of the modes{}'.format(i + 1))
    plt.tight_layout();
    plt.show()


    # 虚部频谱
    for i in range(K):
        plt.subplot(K, 2, 2*i + 1 + 1 )
        plt.plot(np.abs(u_hat_real[:, i]))
        plt.title('（VMD）amplitude frequency of the modes{}'.format(i + 1))
    plt.tight_layout();
    plt.show()

    return u_real,u_imag,u_hat_real,u_hat_imag

