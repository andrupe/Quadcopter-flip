/*Imports Required By Crazyflie*/

#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include "app.h"
#include "controller_pid.h"
#include "FreeRTOS.h"
#include "task.h"
#define DEBUG_MODULE "RLCONTROLLER"
#include "debug.h"
#include "power_distribution.h"
#include "math.h"

/*Custom Libraries*/
#include "prototypes.h"
#include "matrices.h"

/*Initialize neural network parameters*/
NN nn = {
    .obs_mean = OBS_MEAN,
    .obs_std  = OBS_STD,
    .hidden0_w = hidden_0_weights,
    .hidden0_b = hidden_0_bias,
    .hidden1_w = hidden_1_weights,
    .hidden1_b = hidden_1_bias,
};
static unsigned int debug_print_counter = 0; 

void appMain() {
  DEBUG_PRINT("Waiting for activation ...\n");

  while(1) 
  {
    vTaskDelay(M2T(2000));
    // DEBUG_PRINT("Hello from custom controller!\n");
  }
}
/*
  Declaration of global structs 
*/
  static float current_state_data[STATE_SIZE];
  static float obs_data[OBS_SIZE];
  static float z1_data[64];
  static float z2_data[64];
  static float z3_data[OBS_SIZE];
  static float output_data[8];
  static float actions_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  static float action_history_data[4 * ACT_HIST_LEN];
  static float R_wb_data[9];
  static float R_bw_data[9];
  static float R_x_data[9];
  static float R_y_data[9];
  static float R_z_data[9];
  static float R_tmp_data[9];
  static float quat_data[4];
  

  Matrix current_state  = { .rows = 18,               .cols = 1, .data = current_state_data  };
  Matrix obs            = { .rows = OBS_SIZE,         .cols = 1, .data = obs_data            };
  Matrix z1             = { .rows = 64,               .cols = 1, .data = z1_data             };
  Matrix z2             = { .rows = 64,               .cols = 1, .data = z2_data             };
  Matrix z3             = { .rows = OBS_SIZE,         .cols = 1, .data = z3_data             };
  Matrix output         = { .rows = 8,                .cols = 1, .data = output_data         };
  Matrix actions        = { .rows = 4,                .cols = 1, .data = actions_data        };
  Matrix action_history = { .rows = 4 * ACT_HIST_LEN, .cols = 1, .data = action_history_data };
  Matrix R_wb           = { .rows = 3,                .cols = 3, .data = R_wb_data           };
  Matrix R_bw           = { .rows = 3,                .cols = 3, .data = R_bw_data           };
  Matrix R_x            = { .rows = 3,                .cols = 3, .data = R_x_data            };
  Matrix R_y            = { .rows = 3,                .cols = 3, .data = R_y_data            };
  Matrix R_z            = { .rows = 3,                .cols = 3, .data = R_z_data            };
  Matrix R_tmp          = { .rows = 3,                .cols = 3, .data = R_tmp_data          };

  Matrix quat          = { .rows = 4,                 .cols = 1, .data = quat_data           };


void controllerOutOfTreeInit()
{
  /* Set Everything to zero. */
  memset(current_state_data,  0, sizeof(current_state_data));
  memset(obs_data,            0, sizeof(obs_data));
  memset(z1_data,             0, sizeof(z1_data));
  memset(z2_data,             0, sizeof(z2_data));
  memset(z3_data,             0, sizeof(z3_data));
  memset(output_data,         0, sizeof(output_data));
  memset(actions_data,        0, sizeof(actions_data));
  memset(action_history_data, 0, sizeof(action_history_data));
  memset(R_wb_data,           0, sizeof(R_wb_data));
  memset(R_bw_data,           0, sizeof(R_bw_data));
  memset(R_x_data,            0, sizeof(R_x_data));
  memset(R_y_data,            0, sizeof(R_y_data));
  memset(R_z_data,            0, sizeof(R_z_data));
  memset(R_tmp_data,          0, sizeof(R_tmp_data));
  memset(quat_data,           0, sizeof(quat_data));

  DEBUG_PRINT("CONTROLLER INITIALIZED!");

  return;
}

bool controllerOutOfTreeTest()
{
  return true;
}

float body_torque_x = 0.f; 
float body_torque_y = 0.f;
float body_torque_z = 0.f;
float total_thrust  = 0.f;

void controllerOutOfTree( control_t *control, 
                          const setpoint_t *setpoint, 
                          const sensorData_t *sensors,
                          const state_t *state,
                          const uint32_t tick ) 
{
 if (RATE_DO_EXECUTE(RATE_HL_COMMANDER, tick)) 
  {
    // // Store Current Position 
      current_state.data[0] = state->position.x;  
      current_state.data[1] = state->position.y;
      current_state.data[2] = state->position.z;

      // Get orientation as quat and convert it to rotation matrix
      quat.data[0] = state->attitudeQuaternion.w;
      quat.data[1] = state->attitudeQuaternion.x;
      quat.data[2] = state->attitudeQuaternion.y;
      quat.data[3] = state->attitudeQuaternion.z;
      quat_to_matrix(&quat, &R_wb);
      transpose(&R_wb, &R_bw);

      // Copy rotation matrix to state vector
      memcpy(&current_state.data[3], R_wb.data, 9 * sizeof(float));
      
     // Convert velocity  to body frame
      current_state.data[12] = R_bw.data[0]*state->velocity.x + R_bw.data[1]*state->velocity.y + R_bw.data[2]*state->velocity.z;
      current_state.data[13] = R_bw.data[3]*state->velocity.x + R_bw.data[4]*state->velocity.y + R_bw.data[5]*state->velocity.z;
      current_state.data[14] = R_bw.data[6]*state->velocity.x + R_bw.data[7]*state->velocity.y + R_bw.data[8]*state->velocity.z;

      
      // Convert gyros to rad
      current_state.data[15]  =  sensors->gyro.x  * (M_PIf / 180.0f);
      current_state.data[16] =   sensors->gyro.y * (M_PIf / 180.0f); 
      current_state.data[17] =  sensors->gyro.z  * ( M_PIf / 180.0f );
      
      // Change the setpoint (return to zero)
      obs.data[0] = current_state.data[0] - setpoint -> position.x;
      obs.data[1] = current_state.data[1] - setpoint -> position.y;
      obs.data[2] = current_state.data[2] - setpoint -> position.z;

        // obs.data[0] = current_state.data[0];
        // obs.data[1] = current_state.data[1];
        // obs.data[2] = current_state.data[2];
        
        // Copy 
      memcpy(&obs.data[3], &current_state.data[3], 9 * sizeof(float));
      memcpy(&obs.data[12], &current_state.data[12], 3 * sizeof(float));
      memcpy(&obs.data[15], &current_state.data[15], 3 * sizeof(float));
      memcpy(&obs.data[18], action_history.data, 4 * sizeof(float));

        // Uncomment the function bellow to overwrite all values of obs to specific one for DEBUGING         
        // set_all_obs(&obs, 0.0f);   

        /*Normalize Observations*/
        subtract(&obs, &nn.obs_mean, &z3);
        divide_ew(&z3, &nn.obs_std, &obs);
        
        // Clip
        for (int j = 0; j< OBS_SIZE; j++)
          obs.data[j] = clip(obs.data[j], -10, 10);
        
        forward_pass(&obs, &nn, &output, &z1, &z2);
        memcpy(actions.data, output.data, 4*sizeof(float));
        
        // map and clip +- 40% to  of the hoverthrust
        actions.data[0] = clip(DR_HOVER_RPM + actions.data[0] * 0.4f * DR_HOVER_RPM, 0.f * DR_HOVER_RPM, DR_MAX_THRUST);
        actions.data[1] = clip(DR_HOVER_RPM + actions.data[1] * 0.4f * DR_HOVER_RPM, 0.f * DR_HOVER_RPM, DR_MAX_THRUST);
        actions.data[2] = clip(DR_HOVER_RPM + actions.data[2] * 0.4f * DR_HOVER_RPM, 0.f * DR_HOVER_RPM, DR_MAX_THRUST);
        actions.data[3] = clip(DR_HOVER_RPM + actions.data[3] * 0.4f * DR_HOVER_RPM, 0.f * DR_HOVER_RPM, DR_MAX_THRUST);
        
        action_history.data[0] = output.data[0];
        action_history.data[1] = output.data[1];
        action_history.data[2] = output.data[2];
        action_history.data[3] = output.data[3];

        float u0_sq = actions.data[0] * actions.data[0];
        float u1_sq = actions.data[1] * actions.data[1];
        float u2_sq = actions.data[2] * actions.data[2];
        float u3_sq = actions.data[3] * actions.data[3];

        body_torque_x = DR_L * DR_KF * (-u0_sq - u1_sq + u2_sq + u3_sq ) / sqrtf(2.);
        body_torque_y = DR_L * DR_KF * (-u0_sq + u1_sq + u2_sq - u3_sq ) / sqrtf(2.);
        body_torque_z =        DR_KM * (-u0_sq + u1_sq - u2_sq + u3_sq);
        total_thrust  =        DR_KF * ( u0_sq + u1_sq + u2_sq + u3_sq);

    }

    if ( setpoint->mode.z == modeDisable )
    {
        control->thrustSi = 0.0f;
        control->torqueX =0.0f;
        control->torqueY = 0.0f;
        control->torqueZ = 0.0f;
    }
    else
    {
        control->thrustSi = total_thrust;
        control->torqueX  = body_torque_x;
        control->torqueY  = body_torque_y;
        control->torqueZ  = body_torque_z;
    }
    control->controlMode = controlModeForceTorque;



    /*Printing debug messages*/
    // if (RATE_DO_EXECUTE(1, debug_print_counter))
    // {
    //   debug_print_counter = 0;
    //   // Torques sent to the motors
    //   // DEBUG_PRINT("TT  = %.5f\n",   (double) total_thrust);
    //   // DEBUG_PRINT("BTx = %.5f\n",   (double) body_torque_x);
    //   // DEBUG_PRINT("BTy = %.5f\n",   (double) body_torque_y);
    //   // DEBUG_PRINT("BTz = %.5f\n",   (double) body_torque_z);
    //   // DEBUG_PRINT("\n");
      
    //   //  Observations - State Part
    //   DEBUG_PRINT("x = %.3f\n",     (double) obs.data[0]);
    //   DEBUG_PRINT("y = %.3f\n",     (double) obs.data[1]);
    //   DEBUG_PRINT("z = %.3f\n",     (double) obs.data[2]);

    //   DEBUG_PRINT("R1 = %.3f\n",     (double) obs.data[3]);
    //   DEBUG_PRINT("R2 = %.3f\n",     (double) obs.data[4]);
    //   DEBUG_PRINT("R3 = %.3f\n",     (double) obs.data[5]);
    //   DEBUG_PRINT("R4 = %.3f\n",     (double) obs.data[6]);
    //   DEBUG_PRINT("R5 = %.3f\n",     (double) obs.data[7]);
    //   DEBUG_PRINT("R6 = %.3f\n",     (double) obs.data[8]);
    //   DEBUG_PRINT("R7 = %.3f\n",     (double) obs.data[9]);
    //   DEBUG_PRINT("R8 = %.3f\n",     (double) obs.data[10]);
    //   DEBUG_PRINT("R9 = %.3f\n",     (double) obs.data[11]);

    //   // DEBUG_PRINT("vx = %.3f\n",     (double) obs.data[12]);
    //   // DEBUG_PRINT("vy = %.3f\n",     (double) obs.data[13]);
    //   // DEBUG_PRINT("vz = %.3f\n",     (double) obs.data[14]);
    //   // DEBUG_PRINT("wx = %.3f\n",     (double) obs.data[15]);
    //   // DEBUG_PRINT("wy = %.3f\n",     (double) obs.data[16]);
    //   // DEBUG_PRINT("wz = %.3f\n",     (double) obs.data[17]);

    //   DEBUG_PRINT("rad1 = %.3f\n",     (double) actions.data[0]);
    //   DEBUG_PRINT("rad2 = %.3f\n",     (double) actions.data[1]);
    //   DEBUG_PRINT("rad3 = %.3f\n",     (double) actions.data[2]);
    //   DEBUG_PRINT("rad4 = %.3f\n",     (double) actions.data[3]);
    //   // // // current actions
    //   DEBUG_PRINT("a1 = %.5f\n", (double) output.data[0]);
    //   DEBUG_PRINT("a2 = %.5f\n", (double) output.data[1]);
    //   DEBUG_PRINT("a3 = %.5f\n", (double) output.data[2]);
    //   DEBUG_PRINT("a4 = %.5f\n", (double) output.data[3]);
    //   // DEBUG_PRINT("\n");

    //   // // Current setpoint
    //   // DEBUG_PRINT("setpoint x = %.3f, y = %.3f, z= %.3f\n", (double) setpoint->position.x,(double) setpoint->position.y,(double) setpoint->position.z);
    // }
    
    debug_print_counter += 1;
  return;
}

/* Functions used above*/
float clip(float val, float min_val, float max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

void subtract(Matrix *A, Matrix *B, Matrix *C) 
{
    for (int i = 0; i < A->rows * A->cols; i++) 
        C->data[i] = A->data[i] - B->data[i];   
}

void divide_ew(Matrix *A, Matrix *B, Matrix *C) 
{
    for (int i = 0; i < A->rows * A->cols; i++) 
        C->data[i] = A->data[i] / B->data[i];   
}

void multiply(Matrix *A, Matrix *B, Matrix *C)
{
    for(int i = 0; i < A->rows; i++)
    {
        for (int j = 0; j < B->cols; j++)
        {
            float sum = 0;

            for (int k = 0; k < A->cols; k++)
            {
              sum += A->data[i * A->cols + k] * B->data[k * B->cols + j];
            }
              C->data[i * C->cols + j] = sum;
        }
    }
}

void transpose(Matrix *A, Matrix *C) 
{
    for(int i = 0; i < A->rows; i++) 
    {
        for (int j = 0; j < A->cols; j++) 
        {
            C->data[j * C->cols + i] = A->data[i * A->cols + j];
        }
    }
}

void set_all_obs (Matrix *obs, float value)
{
  for (int i = 0; i < OBS_SIZE; i++)
    obs -> data[i] = value;
}

void quat_to_matrix(Matrix *quat, Matrix *R)
{
    float w = quat -> data[0], x = quat -> data[1], y = quat -> data[2], z = quat -> data[3];

    R->data[0] = w * w + x * x - y * y - z * z;
    R->data[1] = 2.0f * (x * y - w * z);
    R->data[2] = 2.0f * (x * z + w * y);

    R->data[3] = 2.0f * (x * y + w * z);
    R->data[4] = w * w - x * x + y * y - z * z;
    R->data[5] = 2.0f * (y * z - w * x);

    R->data[6] = 2.0f * (x * z - w * y);
    R->data[7] = 2.0f * (y * z + w * x);
    R->data[8] = w * w - x * x - y * y + z * z;
}
float SiLUn(float x)
{
    return x / (1 + (expf(-x)));
}   

void multiply_add_activate(Matrix *x1, Matrix *weight, Matrix *bias, Matrix *C, float (*activation)(float))
{
    // activation(weight*x1 + bias)

    float sum = 0;
    int counter = 0;
    register int i, j;

    for (i = 0; i < weight->rows; i++)
    {
        sum = 0;
        for (j = 0; j < weight->cols; j++)
            sum += weight->data[counter++] * x1->data[j];
        C->data[i] = activation(sum + bias->data[i]);

    }
}
void forward_pass(Matrix *input, NN *nn, Matrix *output,  Matrix *z1, Matrix *z2)
{
    multiply_add_activate(input, &nn->hidden0_w, &nn->hidden0_b, z1, SiLUn);
    multiply_add_activate(z1, &nn->hidden1_w, &nn->hidden1_b, output, tanhf);    
}
