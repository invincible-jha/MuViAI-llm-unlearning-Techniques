import matplotlib.pyplot as plt
import numpy as np

def plot_metric(metric_values, metric_name, before_label='Before Unlearning', after_label='After Unlearning'):
    epochs = np.arange(1, len(metric_values[before_label]) + 1)
    plt.plot(epochs, metric_values[before_label], label=before_label)
    plt.plot(epochs, metric_values[after_label], label=after_label)
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Before and After Unlearning')
    plt.legend()
    plt.show()

def analyze_results(results):
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'loss']
    for metric in metrics:
        plot_metric(results[metric], metric.capitalize())

def compare_in_out_distribution(in_distribution_results, out_distribution_results):
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'loss']
    for metric in metrics:
        plt.figure()
        plt.plot(in_distribution_results[metric], label='In-Distribution')
        plt.plot(out_distribution_results[metric], label='Out-Distribution')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.title(f'In-Distribution vs Out-Distribution {metric.capitalize()}')
        plt.legend()
        plt.show()

def generate_insights(results):
    insights = {}
    for metric, values in results.items():
        before_avg = np.mean(values['Before Unlearning'])
        after_avg = np.mean(values['After Unlearning'])
        insights[metric] = {
            'Before Unlearning': before_avg,
            'After Unlearning': after_avg,
            'Difference': after_avg - before_avg
        }
    return insights

if __name__ == "__main__":
    # Example results dictionary
    results = {
        'accuracy': {
            'Before Unlearning': [0.85, 0.86, 0.87],
            'After Unlearning': [0.83, 0.84, 0.85]
        },
        'precision': {
            'Before Unlearning': [0.80, 0.81, 0.82],
            'After Unlearning': [0.78, 0.79, 0.80]
        },
        'recall': {
            'Before Unlearning': [0.75, 0.76, 0.77],
            'After Unlearning': [0.73, 0.74, 0.75]
        },
        'f1_score': {
            'Before Unlearning': [0.77, 0.78, 0.79],
            'After Unlearning': [0.75, 0.76, 0.77]
        },
        'loss': {
            'Before Unlearning': [0.40, 0.38, 0.36],
            'After Unlearning': [0.42, 0.40, 0.38]
        }
    }

    analyze_results(results)

    # Example in-distribution and out-distribution results
    in_distribution_results = {
        'accuracy': [0.85, 0.86, 0.87],
        'precision': [0.80, 0.81, 0.82],
        'recall': [0.75, 0.76, 0.77],
        'f1_score': [0.77, 0.78, 0.79],
        'loss': [0.40, 0.38, 0.36]
    }

    out_distribution_results = {
        'accuracy': [0.83, 0.84, 0.85],
        'precision': [0.78, 0.79, 0.80],
        'recall': [0.73, 0.74, 0.75],
        'f1_score': [0.75, 0.76, 0.77],
        'loss': [0.42, 0.40, 0.38]
    }

    compare_in_out_distribution(in_distribution_results, out_distribution_results)

    insights = generate_insights(results)
    for metric, insight in insights.items():
        print(f"{metric.capitalize()} - Before: {insight['Before Unlearning']}, After: {insight['After Unlearning']}, Difference: {insight['Difference']}")
