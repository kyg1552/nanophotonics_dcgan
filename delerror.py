import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils import createDirectory, delerrordata

root = '../dataset'
spectrum_z = 200

error_spectrum_path = os.path.join(root, 'error/spectrum')
error_pattern_path = os.path.join(root, 'error/pattern')
createDirectory(error_spectrum_path)
createDirectory(error_pattern_path)

spectrum_file_path = os.path.join(root, 'spectrum')
pattern_file_path = os.path.join(root, 'pattern')

spectrumset = delerrordata(
    spectrum_path=spectrum_file_path, 
    pattern_path=pattern_file_path, 
    error_spectrum_path=error_spectrum_path,
    error_pattern_path=error_pattern_path,
    spectrum_z=spectrum_z
)


