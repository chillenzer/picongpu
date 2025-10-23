import matplotlib.pyplot as plt


def visualise_one(values, name):
    plt.figure()
    plt.title(name)
    print(values.shape)
    plt.imshow(values[32, :, :])
    plt.colorbar()


def plot_some(values, name):
    plt.figure()
    plt.title(name)
    for value in values:
        plt.plot(value[32, :, 16])


def visualise(lhs, rhs, lhs_name="lhs", rhs_name="rhs"):
    visualise_one(lhs, lhs_name)
    visualise_one(rhs, rhs_name)
    visualise_one(lhs - rhs, f"{lhs_name} - {rhs_name}")
    visualise_one(lhs / rhs, f"{lhs_name} / {rhs_name}")
    plt.show()
