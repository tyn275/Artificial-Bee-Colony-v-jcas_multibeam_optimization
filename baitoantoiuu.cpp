// Trien khai thuat toan Artificial Bee Colony 

#include <bits/stdc++.h>
#include <complex>

using namespace std;

// --- Cau hinh chung ---
const int M = 16; // So luong ang-ten 
const int K = 160; // So luong diem lay mau (sampling points) cho array response
const double PI = 3.141592653589793;

const int COLONY_SIZE = 40;   // Kich thuoc quan the 
const int N_SOURCES = COLONY_SIZE / 2; // So nguon thuc an (SN)
const int MAX_CYCLES = 500;   // So chu ky toi da
const int LIMIT = 50;         // Nguong Trial (limit)

complex<double> Mat_R2[M][M];      // Ma tran R2 = (DA)^H * (DA) (M x M)
complex<double> Vec_Aux[M];        // Vector tu so: (DA)^H * D * v

// Cau truc nguon thuc an (FoodSource)
struct FoodSource{
    complex<double> sol[M];  // Giai phap
    double cost;    // Gia tri ham muc tieu 
    double fitness; // Gia tri fitness
    int trial;      // So lan khong cai thien
};

// Ham tinh dap ung cua steering vector tai mot goc theta cu the
double get_reference_gain(double theta_deg) {
    complex<double> t(0, 0);
    double theta_rad = theta_deg * PI / 180.0;
    double sin_th = sin(theta_rad);
    
    for (int m = 0; m < M; m++) {
        // a(theta)^T * w_ref (voi w_ref la steering vector tai 0 do)
        // w_ref[m] = conj(exp(j * pi * m * sin(0))) = 1.0
        double phase = PI * (double)m * sin_th;
        t += complex<double>(cos(phase), sin(phase));
    }
    return norm(t); // Tra ve |Aw|^2
}

// Ham muc tieu can cuc tieu: f(w) = - [w^H * R1 * w] / [w^H * R2 * w]
// Voi R1 = u * u^H, R2 da duoc tinh truoc.
double calculate_cost (complex<double> w[M]){
    complex<double> numerator_term = complex<double>(0.0, 0.0);
    complex<double> denominator_term = complex<double>(0.0, 0.0);

    // 1. Tinh Tu so (Numerator) = |w^H * u|^2
    // Tinh t = w^H * u (Tich vo huong)
    complex<double> t = complex<double>(0.0, 0.0);
    for(int i = 0; i < M; i++) {
        // w[i] * conj(Vec_Aux[i]) (phep nhan phuc)
        t += conj(w[i]) * Vec_Aux[i]; 
    }

    // Tu so = |t|^2 = norm(t)
    numerator_term = norm(t); // norm(t) tra ve |t|^2

    // 2. Tinh Mau so (Denominator) = w^H * R2 * w
    // Tinh b = R2 * w (Tich ma tran-vector: M x M * M x 1 = M x 1)
    complex<double> b[M];
    for(int i = 0; i < M; i++) {
         b[i] = complex<double>(0.0, 0.0);
        for(int j = 0; j < M; j++) {
            b[i] += Mat_R2[i][j] * w[j];
        }
    }

    //Mau so = w^H * b (Tich vo huong)
    complex<double> d = complex<double>(0.0, 0.0);
    for(int i = 0; i < M; i++) {
        // conj(w[i]) * b[i]
        d += conj(w[i]) * b[i];
    }   
    // Mau so phai la so thuc vi w^H * R2 * w la dang Hermitian
    denominator_term = d.real(); 

    // 3. Tinh Cost
    if (denominator_term.real() < 1e-12) {
        // Tranh chia cho 0
    return -1.0; 
    }

    // cost = - (Tu so / Mau so). Vi cost la double nen phai lay real()
    return - (numerator_term.real() / denominator_term.real());
}

// Ham tinh fitness 
double calculate_fitness(double cost){
    // fit = 1/(1+obj) neu obj >= 0, fit = 1 + |obj| neu obj < 0
    if (cost >= 0)
        return 1.0 / (1.0 + cost);
    else
        return 1.0 + fabs(cost);
}

// --- Ham ngau nhien va gioi han ---

// Sinh so ngau nhien [0, 1]
double rand_0_1() {
    return static_cast<double>(rand()) / RAND_MAX;
}

// Sinh so ngau nhien [-1, 1]
double rand_neg1_1() {
    return 2.0 * rand_0_1() - 1.0;
}

// Sinh so nguyen ngau nhien [min_val, max_val-1]
int rand_int(int min_val, int max_val) {
    return (rand() % (max_val - min_val)) + min_val;
}

// Cat nghiem ve trong mien 
// void clip_to_bounds(double sol[D]) {
//     for (int i = 0; i < D; i++) {
//         sol[i] = max(GLOBAL_BOUNDS[i].lower, min(sol[i], GLOBAL_BOUNDS[i].upper));
//     }
// }

// dam bao w^H.w = 1
void normalize_solution(complex<double> sol[M]) {
    double norm_sq = 0.0;
    for(int i = 0; i < M; i++) {
        norm_sq += norm(sol[i]); // norm tra ve thuc^2 + ao^2
    }
    double magnitude = sqrt(norm_sq);
    
    if (magnitude > 1e-12) {
        for(int i = 0; i < M; i++) {
            sol[i] /= magnitude;
        }
    }
}

// Khoi tao vi tri ngau nhien
void random_food_source(complex<double> sol[M]) {
    for (int i = 0; i < M; i++) {
        double re = rand_neg1_1();
        double im = rand_neg1_1();
        sol[i] = complex<double>(re, im);
    }
    // Chuan hoa
    normalize_solution(sol);
}

// --- Pha khoi tao ---

// Pha khoi tao
void initialize_sources(FoodSource sources[N_SOURCES], FoodSource& best_global_solution, int& n_evals) {
    // Khoi tao hat giong ngau nhien
    // srand(42);
    
    // Khoi tao nguon thuc an dau tien lam best tam thoi
    random_food_source(best_global_solution.sol);
    best_global_solution.cost = calculate_cost(best_global_solution.sol);
    best_global_solution.fitness = calculate_fitness(best_global_solution.cost);
    best_global_solution.trial = 0;
    n_evals = 1; // So lan goi ham muc tieu

    // Lap qua de tao N_SOURCES nguon thuc an
    for (int m = 0; m < N_SOURCES; m++) {
        FoodSource fs; 
        
        random_food_source(fs.sol);

        fs.cost = calculate_cost(fs.sol);
        fs.fitness = calculate_fitness(fs.cost);
        fs.trial = 0;
        
        sources[m] = fs;

        // Cap nhat loi giai tot nhat
        if (fs.cost < best_global_solution.cost) {
            best_global_solution = fs;
        }
    }
    n_evals += N_SOURCES;
}

// --- Ham Lua chon Tham lam (Greedy Selection) ---

// Tim kiem cuc bo + lua chon tham lam
void search_and_greedy_select(FoodSource& current_source, FoodSource sources[N_SOURCES], int& n_evals) {
    int i = &current_source - sources; 

    // Chon k != i
    int k = rand_int(0, N_SOURCES);
    while (k == i) {
        k = rand_int(0, N_SOURCES);
    }

    // Chon chieu j
    int j = rand_int(0, M);

    // phi ngau nhien [-1, 1]
    double phi = rand_neg1_1();

    // Tao ung vien
    // double candidate_sol[D];
    complex<double> candidate_sol[M];
    copy(current_source.sol, current_source.sol + M, candidate_sol);

    // v_ij = x_ij + phi * (x_ij - x_kj)
    candidate_sol[j] = current_source.sol[j] + phi * (current_source.sol[j] - sources[k].sol[j]);
    
    // clip_to_bounds(candidate_sol);
    normalize_solution(candidate_sol);

    // Tinh cost + fitness
    double cand_obj = calculate_cost(candidate_sol);
    n_evals++;
    double cand_fit = calculate_fitness(cand_obj);

    // Lua chon tham lam
    if (cand_obj < current_source.cost) {
        copy(candidate_sol, candidate_sol + M, current_source.sol);
        current_source.cost = cand_obj;
        current_source.fitness = cand_fit;
        current_source.trial = 0;
    } else {
        current_source.trial++;
    }
}

// --- Pha ong tho ---
void employed_bees_phase(FoodSource sources[N_SOURCES], int& n_evals) {
    for (int i = 0; i < N_SOURCES; i++) {
        search_and_greedy_select(sources[i], sources, n_evals);
    }
}

// --- Pha ong quan sat ---
void onlooker_bees_phase(FoodSource sources[N_SOURCES], int& n_evals) {
    double fit_sum = 0.0;
    for (int i = 0; i < N_SOURCES; i++) {
        fit_sum += sources[i].fitness;
    }

    int sources_visited = 0;
    int current_source_idx = 0;

    while (sources_visited < N_SOURCES) {
        double prob = (fit_sum == 0.0) ? (1.0 / N_SOURCES) : (sources[current_source_idx].fitness / fit_sum);

        double r = rand_0_1();

        if (r < prob) {
            search_and_greedy_select(sources[current_source_idx], sources, n_evals);
            sources_visited++;
        }
        
        current_source_idx = (current_source_idx + 1) % N_SOURCES;
    }
}

// --- Pha ong trinh sat ---
void scout_bees_phase(FoodSource sources[N_SOURCES], int& n_evals) {
    for (int i = 0; i < N_SOURCES; i++) {
        if (sources[i].trial >= LIMIT) {
            FoodSource new_fs;
            random_food_source(new_fs.sol);
            new_fs.cost = calculate_cost(new_fs.sol);
            new_fs.fitness = calculate_fitness(new_fs.cost);
            new_fs.trial = 0;
            
            n_evals++;
            sources[i] = new_fs;
        }
    }
}

// --- Ham chinh ---

int main(){
    srand(42);

    //Tinh R2 va Vec_Aux

    // 1. Dinh nghia cac tham so (Vi du: 3dB beamwidth)
    // Gia su quet tu -90 den 90 do, K=160 diem
    const double MIN_THETA = -90.0;
    const double MAX_THETA = 90.0;
    const double THETA_TARGET = 0.0; // Huong giao tiep mong muon

    const double W_MAIN = 10.0;
    const double W_SIDE = 1.0;
    const double MAINLOBE_DB_DROP = 3.0; // Nguong -3dB

    // A. Xay dung Ma tran A (Array Response Matrix - K x M)
    complex<double> Mat_A[K][M]; 
    complex<double> Vec_v[K]; 
    double d_weights[K];
    
    // 1. Xay dung Mat_A va xac dinh Peak tham chieu
    double peak_gain = get_reference_gain(THETA_TARGET);
    double peak_db = 10.0 * log10(peak_gain + 1e-12);

    for (int k = 0; k < K; k++) {
        double theta_deg = MIN_THETA + k * (MAX_THETA - MIN_THETA) / (K - 1);
        double theta_rad = theta_deg * PI / 180.0;
        
        // Tinh A [cite: 165]
        for (int m = 0; m < M; m++) {
            double phase = PI * (double)m * sin(theta_rad);
            Mat_A[k][m] = complex<double>(cos(phase), sin(phase));
        }

        // 2. Xac dinh v va d_weights dua tren -3dB 
        double current_gain_db = 10.0 * log10(get_reference_gain(theta_deg) + 1e-12);
        if (current_gain_db >= (peak_db - MAINLOBE_DB_DROP)) {
            d_weights[k] = W_MAIN;
            Vec_v[k] = complex<double>(1.0, 0.0);
        } else {
            d_weights[k] = W_SIDE;
            Vec_v[k] = complex<double>(0.0, 0.0);
        }
    }

    // 3. Chuan hoa d_weights (d / sqrt(mean(d^2))
    double sum_d_sq = 0;
    for (int k = 0; k < K; k++) sum_d_sq += d_weights[k] * d_weights[k];
    double rms_d = sqrt(sum_d_sq / K);
    for (int k = 0; k < K; k++) d_weights[k] /= rms_d;

    // 4. Tinh Mat_R2 = (DA)^H * (DA) 
    // R2(i,j) = sum_k [ d[k]^2 * conj(A_ki) * A_kj ]
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            Mat_R2[i][j] = 0;
            for (int k = 0; k < K; k++) {
                Mat_R2[i][j] += (d_weights[k] * d_weights[k]) * conj(Mat_A[k][i]) * Mat_A[k][j];
            }
        }
    }

    // 5. Tinh Vec_Aux (u) = (DA)^H * (Dv) 
    // u(i) = sum_k [ d[k]^2 * conj(A_ki) * v_k ]
    for (int i = 0; i < M; i++) {
        Vec_Aux[i] = 0;
        for (int k = 0; k < K; k++) {
            Vec_Aux[i] += (d_weights[k] * d_weights[k]) * conj(Mat_A[k][i]) * Vec_v[k];
        }
    }

    FoodSource sources[N_SOURCES];
    FoodSource best_global_solution;
    int n_evals = 0;

    vector<double> history_best;

    // Khoi tao
    initialize_sources(sources, best_global_solution, n_evals);
    history_best.push_back(best_global_solution.cost);

    cout << "--- Bat dau Thuat toan Artificial Bee Colony (ABC) ---" << endl;
    cout << "Kich thuoc quan the: " << COLONY_SIZE << ", So nguon: " << N_SOURCES << ", Max Cycles: " << MAX_CYCLES << endl;

    for (int cycle = 1; cycle <= MAX_CYCLES; cycle++) {
        employed_bees_phase(sources, n_evals);

        onlooker_bees_phase(sources, n_evals);

        for (int i = 0; i < N_SOURCES; i++) {
            if (sources[i].cost < best_global_solution.cost) {
                best_global_solution = sources[i];
            }
        }

        scout_bees_phase(sources, n_evals);

        if (history_best.back() > best_global_solution.cost) {
            history_best.push_back(best_global_solution.cost);
        } else {
            history_best.push_back(history_best.back());
        }
    }

    cout << "\n--- Ket qua ABC ---" << endl;
    cout << "Gia tri tot nhat (minimized cost): " << - best_global_solution.cost << endl;
    cout << "Vi tri w_opt tot nhat (w_0, w_1, ...):" << endl;
    for(int i = 0; i < min(M, 16); i++) { 
        cout << "w[" << i << "]: " << best_global_solution.sol[i] << endl;
    }
    cout << "So lan goi ham muc tieu (N_evals): " << n_evals << endl;

    return 0;
}
