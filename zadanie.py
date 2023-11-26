import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl

# Код ядра OpenCL для вычисления функции z = sin(x) * cos(y)
kernel_code = """
__kernel void compute_function(__global float *output, const int width, const int height, const float scale_x, const float scale_y) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        float fx = (float)x / (float)width * scale_x;
        float fy = (float)y / (float)height * scale_y;
        output[y * width + x] = sin(fx) * cos(fy);
    }
}
"""

def main():
    width, height = 200, 200
    scale_x, scale_y = 4.0, 4.0

    # Инициализация OpenCL
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    # Компиляция ядра OpenCL
    program = cl.Program(context, kernel_code).build()

    # Создание буфера для хранения результатов
    output = np.zeros((height, width), dtype=np.float32)
    output_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, output.nbytes)

    # Выполнение ядра OpenCL
    program.compute_function(queue, output.shape, None, output_buffer, np.int32(width), np.int32(height), np.float32(scale_x), np.float32(scale_y))
    cl.enqueue_copy(queue, output, output_buffer).wait()

    # Создание сетки для построения графика
    x_vals = np.linspace(0, scale_x, width)
    y_vals = np.linspace(0, scale_y, height)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Построение графика
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, output, cmap='viridis')
    plt.colorbar(label='z = sin(x) * cos(y)')
    plt.title('график функции  z = sin(x) * cos(y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == "__main__":
    main()
