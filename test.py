import os
import subprocess
#from theano.compat import decode, decode_iter

#location = 'C:\\Users\\smilemango\\AppData\\Local\\Theano\\compiledir_Windows-10-10.0.14393-Intel64_Family_6_Model_60_Stepping_3_GenuineIntel-3.4.5-64\\cuda_ndarray'
#cmd = ['nvcc', '-shared', '-O3', '-Lc:\\Users\\smilemango\\Anaconda3\\envs\\py34\\libs\\', '-LC:\\Users\\smilemango\\Anaconda3\\libs', '-use_fast_math', '--compiler-bindir', 'd:\\MSVS14.0\\VC\\bin\\', '-Xlinker', '/DEBUG', '-D HAVE_ROUND', '-m64', '-Xcompiler', '-D_FORCE_INLINES,-DCUDA_NDARRAY_CUH=mc72d035fdf91890f3b36710688069b2e,-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION,/Zi,/MD', '-IC:\\Users\\smilemango\\Anaconda3\\envs\\py34\\lib\\site-packages\\theano\\sandbox\\cuda', '-IC:\\Users\\smilemango\\Anaconda3\\envs\\py34\\lib\\site-packages\\numpy\\core\\include', '-IC:\\Users\\smilemango\\Anaconda3\\envs\\py34\\include', '-IC:\\Users\\smilemango\\Anaconda3\\envs\\py34\\lib\\site-packages\\theano\\gof', '-o', 'C:\\Users\\smilemango\\AppData\\Local\\Theano\\compiledir_Windows-10-10.0.14393-Intel64_Family_6_Model_60_Stepping_3_GenuineIntel-3.4.5-64\\cuda_ndarray\\cuda_ndarray.pyd', 'C:\\Users\\smilemango\\AppData\\Local\\Theano\\compiledir_Windows-10-10.0.14393-Intel64_Family_6_Model_60_Stepping_3_GenuineIntel-3.4.5-64\\cuda_ndarray\\mod.cu', '-LC:\\Users\\smilemango\\Anaconda3\\envs\\py34\\libs', '-LC:\\Users\\smilemango\\Anaconda3\\envs\\py34', '-lcublas', '-lpython34', '-lcudart']
cmd= ['python','--version']



#os.chdir(location)
p = subprocess.Popen(cmd)
#nvcc_stdout, nvcc_stderr = decode_iter(p.communicate()[:2])
(stdout, stderr) = p.communicate()

print(stderr)
print(p.returncode)