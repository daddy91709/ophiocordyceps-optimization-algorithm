"""
CLI Script to analyze benchmark results from CSV.
"""
import argparse
from src.analysis import load_data, print_summary_statistics, plot_performance_by_dimension


def main():
    parser = argparse.ArgumentParser(description="Analisi dei risultati di benchmark")
    parser.add_argument("--csv", type=str, default="results/risultati.csv", help="Percorso del CSV dei risultati")
    parser.add_argument("--plot", action="store_true", help="Mostra/salva i grafici di analisi")
    parser.add_argument("--save-plot", type=str, default=None, help="Percorso per salvare il grafico (opzionale)")
    args = parser.parse_args()

    try:
        df = load_data(args.csv)
        print_summary_statistics(df)
        if args.plot or args.save_plot:
            plot_performance_by_dimension(df, save_path=args.save_plot)
    except FileNotFoundError as e:
        print(f"Errore: {e}")


if __name__ == "__main__":
    main()
