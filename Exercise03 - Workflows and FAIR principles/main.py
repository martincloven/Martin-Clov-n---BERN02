import argparse
from train_MLP import run_MLP1
from train_MLP_2 import run_MLP2
from test_performance import run_test_performance
from apply_NN import apply_nn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--NeuralNetwork", choices=["1", "2", "both"], default="1")
    args = parser.parse_args()

    if args.NeuralNetwork in ["1", "both"]:
        MLP_model = run_MLP1()
        performance = run_test_performance(MLP_model)
        apply_nn(MLP_model, performance)

    if args.NeuralNetwork in ["2", "both"]:
        MLP_model = run_MLP2()
        performance = run_test_performance(MLP_model)
        apply_nn(MLP_model, performance)

if __name__ == "__main__":
    main()