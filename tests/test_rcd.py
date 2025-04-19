import logging
import sys

sys.path.append("./")

import numpy as np
import pytest
import tomllib
import torch
import torch.nn as nn

from src.rcd import RandomizedCoordinateDescent

# setup config and logging
logger = logging.getLogger(__name__)
with open("tests/config/test_rcd.toml", "rb") as f:
    config = tomllib.load(f)


class QuadraticFunction:
    def __init__(self, A, b):
        if not isinstance(A, torch.Tensor):
            raise TypeError("A must be a torch.Tensor")
        if not isinstance(b, torch.Tensor):
            raise TypeError("b must be a torch.Tensor")
        self.A = A
        self.b = b

    def loss(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        return 0.5 * torch.dot(x, self.A @ x) - torch.dot(self.b, x)

    # def grad(self, x):
    #     if not isinstance(x, torch.Tensor):
    #         raise TypeError("x must be a torch.Tensor")
    #     return self.A @ x - self.b


def test_convergence():
    torch.manual_seed(42)

    d = config["test_convergence"]["dimension"]
    Q = torch.randn(d, d)
    A = Q.T @ Q + 0.1 * torch.eye(d)  # symmetric positive definite matrix
    b = torch.randn(d)

    x_opt = torch.linalg.solve(A, b)  # optimal solution
    f_opt = QuadraticFunction(A, b).loss(x_opt).item()

    x = torch.zeros(d, requires_grad=True)
    optimizer = RandomizedCoordinateDescent([x], lr=config["test_convergence"]["lr"])
    f = QuadraticFunction(A, b)

    losses = []
    num_iters = config["test_convergence"]["iterations"]
    for i in range(num_iters):
        optimizer.zero_grad()
        loss = f.loss(x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item() - f_opt)
        if losses[-1] < config["test_convergence"]["tolerance"]:
            logger.info(f"Early stopped at iteration {i}")
            break

    logger.info("Final loss: %.2e", losses[-1])
    assert (
        losses[-1] < config["test_convergence"]["tolerance"]
    ), "Optimization did not converge sufficiently"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_compatibility():
    torch.manual_seed(42)

    d = config["test_gpu_compatibility"]["dimension"]
    # make everything on CUDA
    Q = torch.randn(d, d, device="cuda")
    A = Q.T @ Q + 0.1 * torch.eye(d, device="cuda")
    b = torch.randn(d, device="cuda")

    # initial x on GPU
    x = torch.zeros(d, device="cuda", requires_grad=True)
    optimizer = RandomizedCoordinateDescent(
        [x], lr=config["test_gpu_compatibility"]["lr"]
    )
    f = QuadraticFunction(A, b)

    num_iters = config["test_gpu_compatibility"]["iterations"]
    initial_x = x.clone().detach()
    for _ in range(num_iters):
        optimizer.zero_grad()
        loss = f.loss(x)
        loss.backward()
        optimizer.step()

    # ensure we moved to GPU and x has been updated
    assert x.device.type == "cuda", "Parameters not on CUDA device"
    assert not torch.allclose(
        x.detach(), initial_x, atol=config["test_gpu_compatibility"]["tolerance"]
    ), "x did not change on GPU"


def test_theoretical_bound():
    torch.manual_seed(42)
    np.random.seed(42)
    d = config["test_theoretical_bound"]["dimension"]
    num_runs = config["test_theoretical_bound"]["runs"]
    num_iters = config["test_theoretical_bound"]["iterations"]
    tol = config["test_theoretical_bound"]["tolerance"]

    all_gaps = np.zeros((num_runs, num_iters))
    for run in range(num_runs):
        Q = torch.randn(d, d)
        A = Q.T @ Q + 0.1 * torch.eye(d)
        b = torch.randn(d)

        x_opt = torch.linalg.solve(A, b)
        f_opt = QuadraticFunction(A, b).loss(x_opt).item()

        eigvals = torch.linalg.eigvalsh(A)
        L = eigvals.max().item()  # lipschitz constant
        mu = eigvals.min().item()  # strong convexity parameter

        x = torch.zeros(d, requires_grad=True)
        optimizer = RandomizedCoordinateDescent([x], lr=1 / L)
        f = QuadraticFunction(A, b)

        for i in range(num_iters):
            optimizer.zero_grad()
            loss = f.loss(x)
            loss.backward()
            optimizer.step()
            gap = loss.item() - f_opt
            all_gaps[run, i] = gap

    mean_gaps = all_gaps.mean(axis=0)
    initial_gap = mean_gaps[0]

    # theorem 5.6
    for i in range(num_iters):
        theoretical = (1 - mu / (d * L)) ** i * initial_gap
        assert (
            mean_gaps[i] <= theoretical + tol
        ), f"Mean empirical gap {mean_gaps[i]} exceeded theoretical bound {theoretical} at iter {i}"


def test_solution_accuracy():
    torch.manual_seed(42)

    d = config["test_solution_accuracy"]["dimension"]
    Q = torch.randn(d, d)
    A = Q.T @ Q + 0.1 * torch.eye(d)
    b = torch.randn(d)
    x_opt = torch.linalg.solve(A, b)

    f = QuadraticFunction(A, b)

    x = torch.zeros(d, requires_grad=True)
    optimizer = RandomizedCoordinateDescent(
        [x], lr=config["test_solution_accuracy"]["lr"]
    )

    for _ in range(config["test_solution_accuracy"]["iterations"]):
        optimizer.zero_grad()
        f.loss(x).backward()
        optimizer.step()

    logger.info("x: %s", x.detach())
    logger.info("x_opt: %s", x_opt)
    assert torch.allclose(
        x.detach(),
        x_opt,
        atol=config["test_solution_accuracy"]["atol"],
        rtol=config["test_solution_accuracy"]["rtol"],
    ), "x not close to x_opt"


@pytest.mark.parametrize(
    "d, A, b, x0, tol, num_iters",
    [
        (1, torch.tensor([[2.0]]), torch.tensor([1.0]), torch.zeros(1), 1e-6, 500),
        (
            2,
            torch.tensor([[3.0, 0.0], [0.0, 1.0]]),
            torch.tensor([1.0, 2.0]),
            torch.zeros(2),
            1e-6,
            500,
        ),
        # you can add more tests here
    ],
)
def test_edge_case_small_dimension(d, A, b, x0, tol, num_iters):
    f = QuadraticFunction(A, b)
    x_opt = torch.linalg.solve(A, b)
    f_opt = f.loss(x_opt).item()

    x = x0.clone().requires_grad_(True)
    optimizer = RandomizedCoordinateDescent(
        [x], lr=config["test_edge_case_small_dimension"]["lr"]
    )

    losses = []
    for _ in range(num_iters):
        optimizer.zero_grad()
        loss = f.loss(x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item() - f_opt)

    logger.info("Final loss: %.2e", losses[-1])
    assert losses[-1] < tol, f"Failed for small dimension (d={d})"


def test_edge_case_optimal_init():
    torch.manual_seed(42)

    d = config["test_edge_case_optimal_init"]["dimension"]
    Q = torch.randn(d, d)
    A = Q.T @ Q + 0.1 * torch.eye(d)
    b = torch.randn(d)

    x_opt = torch.linalg.solve(A, b)
    f = QuadraticFunction(A, b)

    x = x_opt.clone().detach().requires_grad_(True)
    optimizer = RandomizedCoordinateDescent(
        [x], lr=config["test_edge_case_optimal_init"]["lr"]
    )

    optimizer.zero_grad()
    f.loss(x).backward()
    optimizer.step()

    logger.info("x after 1 step: %s", x.detach())
    assert torch.allclose(
        x.detach(), x_opt, atol=config["test_edge_case_optimal_init"]["tolerance"]
    ), "Should remain at optimal"


def test_gradient_consistency():
    torch.manual_seed(42)
    tol = config["test_gradient_consistency"]["tolerance"]

    d = config["test_gradient_consistency"]["dimension"]
    Q = torch.randn(d, d, dtype=torch.double)
    A = Q.T @ Q + 0.1 * torch.eye(d, dtype=torch.double)
    b = torch.randn(d, dtype=torch.double)

    f = QuadraticFunction(A, b)

    x = torch.randn(d, dtype=torch.double, requires_grad=True)
    loss = f.loss(x)
    loss.backward()
    reverse_grad = x.grad.clone()  # gradient from backward

    # compute jacobian, this should be equal to the gradient
    jacobian = torch.autograd.functional.jacobian(f.loss, x).reshape(d)
    assert torch.allclose(jacobian, reverse_grad, rtol=tol, atol=tol)

    # additional check with vector product
    for _ in range(3):
        x_fwd = x.detach().clone().requires_grad_(True)
        tangent = torch.randn(d, dtype=torch.double)
        _, jvp = torch.autograd.functional.jvp(f.loss, x_fwd, tangent)
        dot_product = torch.dot(reverse_grad, tangent)
        assert abs(jvp.item() - dot_product.item()) < tol

    # compute hessian
    def compute_grad(x):
        x_copy = x.clone().detach().requires_grad_(True)
        f.loss(x_copy).backward()
        return x_copy.grad

    hessian = torch.autograd.functional.jacobian(
        compute_grad, x.detach().clone().requires_grad_(True)
    )
    # hessian should be symmetric for twice differentiable function
    assert torch.allclose(hessian, hessian.T, rtol=tol, atol=tol)


@pytest.mark.xfail(
    reason="Convergence may fail due to ill conditioning. Try increasing the number of iterations or decreasing the learning rate."
)
def test_outlier_values():
    torch.manual_seed(42)

    d = config["test_outlier_values"]["dimension"]
    diag_values = torch.ones(d)
    diag_values[0] = config["test_outlier_values"]["diag0"]  # small value
    diag_values[1] = config["test_outlier_values"]["diag1"]  # large value
    A = torch.diag(diag_values)  # ill conditioned matrix
    b = torch.randn(d) * 10

    f = QuadraticFunction(A, b)
    x_opt = torch.linalg.solve(A, b)
    f_opt = f.loss(x_opt).item()

    x = torch.zeros(d, requires_grad=True)
    # L = diag_values.max().item()
    optimizer = RandomizedCoordinateDescent([x], lr=config["test_outlier_values"]["lr"])

    num_iters = config["test_outlier_values"]["iterations"]
    for i in range(num_iters):
        optimizer.zero_grad()
        loss = f.loss(x)
        if torch.isnan(loss) or torch.isinf(loss):
            pytest.fail(f"Loss became NaN or Inf at iteration {i}")
        loss.backward()
        optimizer.step()
        if abs(f.loss(x).item() - f_opt) < config["test_outlier_values"]["tolerance"]:
            break

    final_gap = abs(f.loss(x).item() - f_opt)
    logger.info("Final loss gap: %.2e", final_gap)
    assert (
        final_gap < config["test_outlier_values"]["tolerance"]
    ), "Did not converge with outlier values"


# def generate_friedman1(
#     n_samples: int = 200, n_features: int = 10, noise_std: float = 1.0
# ):
#     X = np.random.rand(n_samples, n_features)
#     noise = np.random.randn(n_samples) * noise_std
#     y = (
#         10 * np.sin(np.pi * X[:, 0] * X[:, 1])
#         + 20 * ((X[:, 2] - 0.5) ** 2)
#         + 10 * X[:, 3]
#         + 5 * X[:, 4]
#         + noise
#     )
#     return X, y


@pytest.mark.parametrize(
    "n_samples,n_features,noise_std",
    [
        (200, 10, 1.0),
        (300, 10, 0.5),
        (400, 10, 0.1),
        # you can add more tests here
    ],
)
def test_friedman_dataset(n_samples, n_features, noise_std):
    from sklearn.datasets import make_friedman1

    np.random.seed(42)
    torch.manual_seed(42)

    X_np, y_np = make_friedman1(
        n_samples=n_samples, n_features=n_features, noise=noise_std, random_state=42
    )
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)

    # simple linear regression model
    model = nn.Linear(n_features, 1)

    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)

    criterion = nn.MSELoss()

    optimizer = RandomizedCoordinateDescent(
        model.parameters(), lr=config["test_friedman_dataset"]["lr"]
    )

    losses = []
    num_iters = config["test_friedman_dataset"]["iterations"]
    for i in range(num_iters):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # check if the loss has dropped below fraction of the initial loss
        if loss.item() < config["test_friedman_dataset"]["fraction"] * losses[0]:
            logger.info(f"Pass at iteration {i} with loss {loss.item():.4f}")
            break

    logger.info("Initial loss: %.2e", losses[0])
    logger.info("Final loss: %.2e", losses[-1])
    assert losses[-1] < losses[0], "Loss did not improve on Friedman dataset"


def test_type_errors():
    # wrong type for A and b
    with pytest.raises(TypeError):
        QuadraticFunction(A=[1, 2], b=[3, 4])
    with pytest.raises(TypeError):
        QuadraticFunction(A=torch.eye(2), b=[1, 2])

    # wrong type for x in loss/grad
    A = torch.eye(2)
    b = torch.ones(2)
    f = QuadraticFunction(A, b)
    with pytest.raises(TypeError):
        f.loss([1, 2])
    # with pytest.raises(TypeError):
    #     f.grad([1, 2])

    # wrong shape for x
    with pytest.raises(RuntimeError):
        f.loss(torch.ones(3))
    # with pytest.raises(RuntimeError):
    #     f.grad(torch.ones(3))

    # optimizer param is not tensor
    with pytest.raises(TypeError):
        RandomizedCoordinateDescent([123], lr=0.1)

    # invalid learning rate
    x = torch.zeros(2, requires_grad=True)
    with pytest.raises(ValueError):
        RandomizedCoordinateDescent([x], lr=-0.1)


def test_empty_param():
    with pytest.raises(Exception):
        RandomizedCoordinateDescent([], lr=0.1)


def test_multiple_parameter_tensor():
    torch.manual_seed(42)

    # two quadratic functions, each with their own parameter
    d1, d2 = 5, 3
    Q1 = torch.randn(d1, d1)
    Q2 = torch.randn(d2, d2)
    A1 = Q1.T @ Q1 + 0.1 * torch.eye(d1)
    b1 = torch.randn(d1)
    A2 = Q2.T @ Q2 + 0.1 * torch.eye(d2)
    b2 = torch.randn(d2)

    x1 = torch.zeros(d1, requires_grad=True)
    x2 = torch.zeros(d2, requires_grad=True)

    f1 = QuadraticFunction(A1, b1)
    f2 = QuadraticFunction(A2, b2)

    def total_loss():
        return f1.loss(x1) + f2.loss(x2)

    optimizer = RandomizedCoordinateDescent(
        [x1, x2], lr=config["test_multiple_parameter_tensor"]["lr"]
    )
    num_iters = config["test_multiple_parameter_tensor"]["iterations"]
    for _ in range(num_iters):
        optimizer.zero_grad()
        loss = total_loss()
        loss.backward()
        optimizer.step()

    x1_opt = torch.linalg.solve(A1, b1)
    x2_opt = torch.linalg.solve(A2, b2)
    assert torch.allclose(
        x1.detach(), x1_opt, atol=config["test_multiple_parameter_tensor"]["tolerance"]
    )
    assert torch.allclose(
        x2.detach(), x2_opt, atol=config["test_multiple_parameter_tensor"]["tolerance"]
    )


def test_reproducibility():
    d = config["test_reproducibility"]["dimension"]
    Q = torch.randn(d, d)
    A = Q.T @ Q + 0.1 * torch.eye(d)
    b = torch.randn(d)
    f = QuadraticFunction(A, b)

    # check if with same seed we get the same result
    def run_once():
        torch.manual_seed(config["test_reproducibility"]["seed"])
        x = torch.zeros(d, requires_grad=True)
        optimizer = RandomizedCoordinateDescent(
            [x], lr=config["test_reproducibility"]["lr"]
        )
        losses = []
        num_iters = config["test_reproducibility"]["iterations"]
        for _ in range(num_iters):
            optimizer.zero_grad()
            loss = f.loss(x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return x.detach().clone(), losses

    x1, losses1 = run_once()
    x2, losses2 = run_once()

    logger.info("Final losses: %s, %s", losses1[-1], losses2[-1])
    assert torch.allclose(
        x1, x2, atol=config["test_reproducibility"]["tolerance"]
    ), "Final parameters differ with same seed"
    assert all(
        abs(a - b) < config["test_reproducibility"]["tolerance"]
        for a, b in zip(losses1, losses2)
    ), "Loss trajectories differ with same seed"


def test_coordinate_wise_update():
    torch.manual_seed(42)

    d = config["test_coordinate_wise_update"]["dimension"]
    Q = torch.randn(d, d)
    A = Q.T @ Q + 0.1 * torch.eye(d)
    b = torch.randn(d)

    f = QuadraticFunction(A, b)

    x = torch.zeros(d, requires_grad=True)
    optimizer = RandomizedCoordinateDescent(
        [x], lr=config["test_coordinate_wise_update"]["lr"]
    )

    num_iters = config["test_coordinate_wise_update"]["iterations"]
    for _ in range(num_iters):
        x_prev = x.detach().clone()
        optimizer.zero_grad()
        f.loss(x).backward()
        optimizer.step()
        x_new = x.detach()
        diff = x_new - x_prev

        # count number of coordinates that changed
        num_changed = (diff.abs() > 1e-8).sum().item()
        assert num_changed == 1, f"More than one coordinate changed: {diff}"


def test_lr_zero():
    torch.manual_seed(42)

    d = config["test_lr_zero"]["dimension"]
    Q = torch.randn(d, d)
    A = Q.T @ Q + 0.1 * torch.eye(d)
    b = torch.randn(d)

    f = QuadraticFunction(A, b)

    x = torch.randn(d, requires_grad=True)
    x_initial = x.detach().clone()
    optimizer = RandomizedCoordinateDescent([x], lr=0.0)

    num_iters = config["test_lr_zero"]["iterations"]
    for _ in range(num_iters):
        optimizer.zero_grad()
        f.loss(x).backward()
        optimizer.step()

        # x should not change
        assert torch.allclose(
            x.detach(), x_initial, atol=config["test_lr_zero"]["tolerance"]
        ), f"Parameters changed despite lr=0.0: {x.detach()} vs {x_initial}"


def test_compare_cifar10():
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms

    batch_size = 32
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )

    # simple model: flatten and linear
    def get_model():
        return nn.Sequential(nn.Flatten(), nn.Linear(32 * 32 * 3, 10))

    def train_on_batch(model, optimizer, images, labels, criterion, num_iters):
        for _ in range(num_iters):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        return criterion(model(images), labels).item()

    # get a single batch
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    num_iters = config["test_compare_cifar10"]["iterations"]
    lr = config["test_compare_cifar10"]["lr"]
    criterion = nn.CrossEntropyLoss()

    # test with RCD
    model_rcd = get_model()
    loss0_rcd = criterion(model_rcd(images), labels).item()
    optimizer_rcd = RandomizedCoordinateDescent(model_rcd.parameters(), lr=lr)
    loss1_rcd = train_on_batch(
        model_rcd, optimizer_rcd, images, labels, criterion, num_iters
    )

    # test with Adam
    model_adam = get_model()
    loss0_adam = criterion(model_adam(images), labels).item()
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=lr)
    loss1_adam = train_on_batch(
        model_adam, optimizer_adam, images, labels, criterion, num_iters
    )

    # test with SGD
    model_sgd = get_model()
    loss0_sgd = criterion(model_sgd(images), labels).item()
    optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=lr)
    loss1_sgd = train_on_batch(
        model_sgd, optimizer_sgd, images, labels, criterion, num_iters
    )

    logger.info(f"RCD: Initial={loss0_rcd:.4f}, Final={loss1_rcd:.4f}")
    logger.info(f"Adam: Initial={loss0_adam:.4f}, Final={loss1_adam:.4f}")
    logger.info(f"SGD: Initial={loss0_sgd:.4f}, Final={loss1_sgd:.4f}")

    assert loss1_rcd < loss0_rcd, "RCD loss did not decrease"
    assert loss1_adam < loss0_adam, "Adam loss did not decrease"
    assert loss1_sgd < loss0_sgd, "SGD loss did not decrease"


def test_compare_mnist():
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms

    batch_size = 32
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )

    def get_model():
        return nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))

    def train_on_batch(model, optimizer, images, labels, criterion, num_iters):
        for _ in range(num_iters):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        return criterion(model(images), labels).item()

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    num_iters = config["test_compare_mnist"]["iterations"]
    lr = config["test_compare_mnist"]["lr"]
    criterion = nn.CrossEntropyLoss()

    # test with RCD
    model_rcd = get_model()
    loss0_rcd = criterion(model_rcd(images), labels).item()
    optimizer_rcd = RandomizedCoordinateDescent(model_rcd.parameters(), lr=lr)
    loss1_rcd = train_on_batch(
        model_rcd, optimizer_rcd, images, labels, criterion, num_iters
    )

    # test with Adam
    model_adam = get_model()
    loss0_adam = criterion(model_adam(images), labels).item()
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=lr)
    loss1_adam = train_on_batch(
        model_adam, optimizer_adam, images, labels, criterion, num_iters
    )

    # test with SGD
    model_sgd = get_model()
    loss0_sgd = criterion(model_sgd(images), labels).item()
    optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=lr)
    loss1_sgd = train_on_batch(
        model_sgd, optimizer_sgd, images, labels, criterion, num_iters
    )

    logger.info(f"RCD:  Initial={loss0_rcd:.4f}, Final={loss1_rcd:.4f}")
    logger.info(f"Adam: Initial={loss0_adam:.4f}, Final={loss1_adam:.4f}")
    logger.info(f"SGD:  Initial={loss0_sgd:.4f}, Final={loss1_sgd:.4f}")

    assert loss1_rcd < loss0_rcd, "RCD loss did not decrease"
    assert loss1_adam < loss0_adam, "Adam loss did not decrease"
    assert loss1_sgd < loss0_sgd, "SGD loss did not decrease"
