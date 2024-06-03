import cv2
import numpy as np
import matplotlib.pyplot as plt

def spectral_residual_saliency(image):
    print('---image---')
    print(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_float = gray.astype(np.float32)
    fft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    magnitude, angle = cv2.cartToPolar(fft[:, :, 0], fft[:, :, 1])

    # Log spectrum
    log_magnitude = np.log(magnitude.clip(min=1e-5))

    # Spectral residual
    spectral_residual = log_magnitude - cv2.boxFilter(log_magnitude, -1, (3, 3))

    # Compute saliency map
    exp_magnitude = np.exp(spectral_residual)
    spectral_residual_fft = cv2.polarToCart(exp_magnitude, angle)
    combined_fft = np.dstack([spectral_residual_fft[0], spectral_residual_fft[1]])
    inverse_fft = cv2.idft(combined_fft)
    saliency_map = cv2.magnitude(inverse_fft[:, :, 0], inverse_fft[:, :, 1])
    print('-----saliency_map---')
    print(saliency_map)

    # Normalize
    cv2.normalize(saliency_map, saliency_map, 0, 255, cv2.NORM_MINMAX)
    return saliency_map.astype(np.uint8)

# Load image and compute saliency
image = cv2.imread('/home/hp/Projects/homework/test.png')
saliency = spectral_residual_saliency(image)
plt.imshow(saliency, cmap='hot')
plt.title('Spectral Residual Saliency')
plt.show()
