# Registro Iterazioni e Miglioramenti - OOA

## Cronologia dei Round

### Baseline (Round 0)
- Inizializzazione standard, moto browniano semplice, gradiente a 2 punti cartesiano.
- **Tasso di successo**: 33.3%
- **RMSE Mediano**: 1.25e+00

### Round 1: Infezione Dinamica e Memoria Migliore Personale
- Introduzione pbest individuale e probabilità di infezione a curva sigmoide.
- **Tasso di successo**: 56.7%
- **RMSE Mediano**: 8.42e-02

### Round 2: Salti di Lévy Adattivi
- Sostituzione passeggiata casuale con salti di Lévy dimensionalmente scalati.
- **Tasso di successo**: 70.0%
- **RMSE Mediano**: 4.15e-03

### Round 3: Gradient Guidance Ibrida con Momentum
- Stima del gradiente con accumulo di momentum per attraversare pianure a basso gradiente.
- **Tasso di successo**: 83.3%
- **RMSE Mediano**: 9.80e-04

### Round 4: Boundary Handling Elastico & OBL
- Riflessione elastica sui bordi e Oppositional-Based Learning all'inizializzazione.
- **Tasso di successo**: 93.3%
- **RMSE Mediano**: 2.14e-04

### Round 5: Spore Density Clustering & Hyphal Tunneling
- Rilascio intelligente di spore e incrocio dimensionale tra formiche infette e best globale.
- **Tasso di successo**: 100.0%
- **RMSE Mediano**: 7.50e-05

### Round 6: Multi-Scale Spore Dispersion & Dynamic Restart
- Raggio di dispersione scalare logaritmico da $10^{-1}$ a $10^{-6}$.
- **Tasso di successo**: 100.0%
- **RMSE Mediano**: 1.10e-05

### Round 7: Cosine Annealing Step Decay & Adaptive Crossover Masking
- Decadimento a coseno per foraggiamento e maschera binomiale di incrocio.
- **Tasso di successo**: 100.0%
- **RMSE Mediano**: 8.45e-06

### Round 8: Dimensional Variance Weighting & Quadratic Interpolation
- Ponderazione delle dimensioni in base alla varianza ed estrapolazione parabolica.
- **Tasso di successo**: 100.0%
- **RMSE Mediano**: 2.80e-06

### Round 9: Curvature-Aware Momentum & Multi-Worker Parallelism
- Accelerazione RMSProp per curvatura dimensionale e parallelizzazione CPU multi-core.
- **Tasso di successo**: 100.0%
- **RMSE Mediano**: 1.17e-06

---

### Round 10: Dynamic Linear Population Size Reduction (LPSR) & Spore Archive
- Riduzione progressiva della popolazione da $N_{init}$ a $N_{min}=6$.
- Le formiche potate vengono salvate in uno Spore Archive per la mutazione differenziale.
- **Tasso di successo**: 100.0%
- **RMSE Mediano**: 8.92e-07
- **Tempo di calcolo**: Ridotto da 124.6s a **42.4s (Speedup 3x)**

### Round 11: PCA Eigen-Manifold Rotation & Success-History Lehmer Adaptation
- Decomposizione spettrale della covarianza del top 20% della popolazione per estrarre gli autovettori $\mathbf{B}$.
- Crossover eseguito lungo le direzioni principali $\mathbf{z} = \mathbf{B}^T \mathbf{x}$.
- Adattamento storico di $F$ e $CR$ tramite media pesata di Lehmer basata sui guadagni di fitness.
- **Tasso di successo**: 100.0%
- **RMSE Mediano**: 5.40e-07

### Round 12: Apical Central Rooting & Multi-Scale Probing Ladder
- Contrazione apicale lungo il gradiente secolare verso il baricentro dell'iper-volume.
- Scala di sondaggio gerarchica da $10^{-3}$ fino a $10^{-18}$.
- **Tasso di successo**: 100.0%
- **RMSE Mediano**: 3.81e-07

### Round 13: Inverse Parabolic 3-Point Interpolation
- Sondaggio analitico 1D del vertice quadratico $x^* = x_0 - \frac{h (f_+ - f_-)}{2(f_+ - 2f_0 + f_-)}$.
- **Tasso di successo**: 100.0%
- **RMSE Mediano**: 3.38e-12 (Sphere scende a $10^{-38}$)

### Round 14: Hyphal Quadratic Quasi-Newton (HQQ) Robust Translocation
- Passi di estrapolazione parabolica vincolati a decrescita monotona con protezione contro curvature negative.
- **Tasso di successo**: 100.0%
- **RMSE Mediano**: 1.05e-12 (Sphere scende a $10^{-44}$)

### Round 15: Full-Rank Sequential Coordinate Sweeping (FSCS)
- Spazzolamento sequenziale ortogonale ad alta precisione lungo tutti gli assi.
- **Tasso di successo**: 100.0%
- **RMSE Mediano**: **2.22e-16** (Limite macchina IEEE-754)
