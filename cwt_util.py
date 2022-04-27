# import numpy as np
# from scipy import signal
# import matplotlib.pyplot as plt
# t = np.linspace(-1, 1, 200, endpoint=False)
# sig  = np.sin(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
# print(sig.shape)
# widths = np.arange(1, 51)
# cwtmatr = signal.cwt(sig, signal.ricker, widths)
# print(cwtmatr.shape)
# plt.imshow(cwtmatr, extent=[-1, 1, 1, 51], cmap='PRGn', aspect='auto',
#            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pywt
from matplotlib.font_manager import FontProperties

# chinese_font = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
sampling_rate = 12000
t = np.arange(0, 1.0, 1.0 / sampling_rate)
f1 = 100
f2 = 200
f3 = 300
data = np.piecewise(t, [t < 1, t < 0.8, t < 0.3],
                    [lambda t: np.sin(2 * np.pi * f1 * t), lambda t: np.sin(2 * np.pi * f2 * t),
                     lambda t: np.sin(2 * np.pi * f3 * t)])
print(data.shape)
wavename = 'cgau8'
totalscal = 256
fc = pywt.central_frequency(wavename)
cparam = 2 * fc * totalscal
scales = cparam / np.arange(totalscal, 1, -1)
[cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)
plt.figure(figsize=(8, 4))
plt.subplot(211)
plt.plot(t, data)
plt.xlabel(u"时间(秒)", )
plt.title(u"300Hz和200Hz和100Hz的分段波形和时频谱",  fontsize=20)
plt.subplot(111)
plt.contourf(t, frequencies, abs(cwtmatr))
plt.ylabel(u"频率(Hz)")
plt.xlabel(u"时间(秒)")
plt.subplots_adjust(hspace=0.4)
plt.show()