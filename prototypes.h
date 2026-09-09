#ifndef INF_HEAD
#define INF_HEAD
typedef struct
{
    int rows;
    int cols;
    float* data;
}Matrix;

typedef struct inference
{
    Matrix obs_mean;
    Matrix obs_std;
    
    Matrix hidden0_w;
    Matrix hidden0_b;

    Matrix hidden1_w;
    Matrix hidden1_b;
}NN;

// Function Prototypes

// Neural Network related
void forward_pass(Matrix * , NN *, Matrix *,  Matrix *, Matrix *);
void multiply_add_activate(Matrix *, Matrix *, Matrix *, Matrix *, float (*activation)(float));
float SiLUn(float);
void subtract(Matrix *, Matrix *, Matrix *);
void divide_ew(Matrix *, Matrix *, Matrix *);

// Matrix Operations
void multiply(Matrix *, Matrix *, Matrix *);
void add(Matrix *, Matrix *, Matrix *);
void transpose(Matrix *A, Matrix *C); 

// Helping functions
float normalize_ang(float );
float clip(float , float , float);
void set_all_obs (Matrix *, float);
void quat_to_matrix(Matrix *, Matrix *);

// Drone Parameters
#define DR_M 0.033f
#define DR_L 0.046f
#define DR_KF 2.25e-8f
#define DR_KM 1.34e-10f
#define DR_G 9.81f
#define DR_HOVER_RPM 1896.5758f // sqrt(G * M / (4 * KF)) rad/s 
// #define DR_HOVER_RPM 1790.5758f // sqrt(G * M / (4 * KF)) rad/s 
// #define DR_HOVER_RPM 2190.0f
#define DR_MAX_THRUST 2617.0f // In rads/
#define M_PIf 3.1415926535f
#define ACT_HIST_LEN 1

#define STATE_SIZE 18 // pos att vell omega
#define OBS_SIZE 22

#endif