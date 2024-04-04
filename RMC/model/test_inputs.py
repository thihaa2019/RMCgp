def cost_tester(cost_func):
    assert hasattr(cost_func,"cost") and callable(cost_func.cost), "Cost is not defined."
    assert hasattr(cost_func,"derivative") and callable(cost_func.derivative), "Derivative is not defined."

def kernel_tester(kernel):
    valid_kernels = ["RBF", "Matern32", "Matern52",None]
    assert kernel in valid_kernels, "Kernel input is not valid."
