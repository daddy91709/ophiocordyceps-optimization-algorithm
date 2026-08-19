"""
Analysis and Visualization Utilities for Benchmark Results.
"""
import os
import numpy as np
import pandas as pd


def load_data(csv_path="results/risultati.csv"):
    """Carica i dati dal CSV gestendo formati ed errori."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File non trovato: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, sep=';')
    
    numeric_cols = ['Best', 'Mean', 'Std', 'Worst', 'RMSE', 'Tempo medio (s)', 'Valore ottimo teorico']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def print_summary_statistics(df):
    """Stampa statistiche riassuntive dei risultati di benchmark."""
    print("=== STATISTICHE RIASSUNTIVE ===")
    total_tests = len(df)
    failed_tests = df[['Mean', 'RMSE']].isna().any(axis=1).sum()
    success_rate = ((total_tests - failed_tests) / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Test totali: {total_tests}")
    print(f"Test falliti (convergenza): {failed_tests}")
    print(f"Tasso di successo: {success_rate:.1f}%\n")
    
    if total_tests > 0 and 'Funzione' in df.columns:
        failed_by_function = df.groupby('Funzione').apply(
            lambda x: x[['Mean', 'RMSE']].isna().any(axis=1).sum()
        ).sort_values(ascending=False)
        
        print("FUNZIONI CON PIÙ PROBLEMI:")
        for func, failures in failed_by_function.head().items():
            total_func_tests = len(df[df['Funzione'] == func])
            print(f"- {func}: {failures}/{total_func_tests} fallimenti")
    
    print("\n" + "=" * 50)
    df_valid = df.dropna(subset=['RMSE']) if 'RMSE' in df.columns else pd.DataFrame()
    if len(df_valid) > 0:
        print("\nSTATISTICHE DATI VALIDI:")
        print(f"RMSE - Media: {df_valid['RMSE'].mean():.2e}")
        print(f"RMSE - Mediana: {df_valid['RMSE'].median():.2e}")
        if 'Tempo medio (s)' in df_valid.columns:
            print(f"Tempo - Media: {df_valid['Tempo medio (s)'].mean():.2f}s")
            print(f"Tempo - Mediana: {df_valid['Tempo medio (s)'].median():.2f}s")


def plot_performance_by_dimension(df, save_path=None):
    """Grafico delle prestazioni in base alle dimensioni."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] Matplotlib non disponibile.")
        return

    df_valid = df.dropna(subset=['Mean', 'RMSE'])
    if len(df_valid) == 0:
        print("Nessun dato valido da plottare.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    dims = sorted(df_valid['Dimensioni'].unique())
    
    axes[0, 0].boxplot([df_valid[df_valid['Dimensioni'] == d]['RMSE'].values for d in dims],
                       labels=dims)
    axes[0, 0].set_title('Distribuzione RMSE per Dimensione')
    axes[0, 0].set_xlabel('Dimensioni')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_yscale('log')
    
    axes[0, 1].boxplot([df_valid[df_valid['Dimensioni'] == d]['Tempo medio (s)'].values for d in dims],
                       labels=dims)
    axes[0, 1].set_title('Tempo di Esecuzione per Dimensione')
    axes[0, 1].set_xlabel('Dimensioni')
    axes[0, 1].set_ylabel('Tempo (s)')
    axes[0, 1].set_yscale('log')
    
    for func in df_valid['Funzione'].unique()[:10]:
        func_data = df_valid[df_valid['Funzione'] == func]
        axes[1, 0].scatter(func_data['Dimensioni'], func_data['RMSE'], label=func, alpha=0.7)
    axes[1, 0].set_xlabel('Dimensioni')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('RMSE vs Dimensioni (prime 10 funzioni)')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    axes[1, 1].scatter(df_valid['Tempo medio (s)'], df_valid['RMSE'], alpha=0.6)
    axes[1, 1].set_xlabel('Tempo medio (s)')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title('Tempo vs RMSE')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Grafico salvato in: {save_path}")
    else:
        plt.show()


def run_complete_analysis(csv_path="results/risultati.csv"):
    """Esegue l'analisi completa dei risultati salvati."""
    print("Caricamento dati...")
    df = load_data(csv_path)
    print_summary_statistics(df)
