// Trien khai thuat toan Artificial Bee Colony 

#include <bits/stdc++.h>

using namespace std;

// --- Cau hinh chung ---
const int D = 2;              // So
const int COLONY_SIZE = 40;   // Kich thuoc quan the 
const int N_SOURCES = COLONY_SIZE / 2; // So nguon thuc an (SN)
const int MAX_CYCLES = 500;   // So chu ky toi da
const int LIMIT = 50;         // Nguong Trial (limit)

// Cau truc de luu tru gioi han (bounds)
struct Bounds {
    double lower; // l_i
    double upper; // u_i
};

// Gia su gioi han la [-5, 5] cho ca hai chieu
const Bounds GLOBAL_BOUNDS[D] = {{ -5.0, 5.0 }, { -5.0, 5.0 }};

// Cau truc nguon thuc an (FoodSource)
struct FoodSource{
    double sol[D];  // Giai phap (vi tri x, y)
    double cost;    // Gia tri ham muc tieu 
    double fitness; // Gia tri fitness
    int trial;      // So lan khong cai thien
};

// Ham muc tieu can cuc tieu 
// f(x, y) = x^4/10 - 2x^3/15 - 2x^2/5 + y^2 + 32/30
double calculate_cost (double sol[D]){
    double x = sol[0];
    double y = sol[1];
    double term1 = pow(x, 4) / 10.0;
    double term2 = (2.0/15.0) * pow(x, 3); 
    double term3 = (2.0/5.0) * pow(x, 2); 
    double term4 = pow(y, 2);
    double term5 = (32.0/30.0); 

    // Gia tri ham muc tieu (Objective Value)
    return term1 - term2 - term3 + term4 + term5;
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
void clip_to_bounds(double sol[D]) {
    for (int i = 0; i < D; i++) {
        sol[i] = max(GLOBAL_BOUNDS[i].lower, min(sol[i], GLOBAL_BOUNDS[i].upper));
    }
}

// Khoi tao vi tri ngau nhien
void random_food_source(double sol[D]) {
    for (int i = 0; i < D; i++) {
        double l_i = GLOBAL_BOUNDS[i].lower;
        double u_i = GLOBAL_BOUNDS[i].upper;
        sol[i] = l_i + rand_0_1() * (u_i - l_i);
    }
}

// --- Pha khoi tao ---

// Pha khoi tao
void initialize_sources(FoodSource sources[N_SOURCES], FoodSource& best_global_solution, int& n_evals) {
    // Khoi tao hat giong ngau nhien
    srand(42);
    
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
    int j = rand_int(0, D);

    // phi ngau nhien [-1, 1]
    double phi = rand_neg1_1();

    // Tao ung vien
    double candidate_sol[D];
    copy(current_source.sol, current_source.sol + D, candidate_sol);

    // v_ij = x_ij + phi * (x_ij - x_kj)
    candidate_sol[j] = current_source.sol[j] + phi * (current_source.sol[j] - sources[k].sol[j]);
    
    clip_to_bounds(candidate_sol);

    // Tinh cost + fitness
    double cand_obj = calculate_cost(candidate_sol);
    n_evals++;
    double cand_fit = calculate_fitness(cand_obj);

    // Lua chon tham lam
    if (cand_obj < current_source.cost) {
        copy(candidate_sol, candidate_sol + D, current_source.sol);
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
    cout << "Gia tri tot nhat: " << best_global_solution.cost << endl;
    cout << "Vi tri [x, y] tot nhat: [" << best_global_solution.sol[0] << ", " << best_global_solution.sol[1] << "]" << endl;
    cout << "So lan goi ham muc tieu (N_evals): " << n_evals << endl;

    return 0;
}
