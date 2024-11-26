#include "main.h"
#include "network_desc.h"
#include "input.h"
#include "weights.h"
#include "output.h"
#include "kernel.h"

int layer_init()
{
#ifdef DEBUG
  printf("-> Entering Layer Initialization...\n");
#endif

  network_layers[0].input_data = input;
  if(network_layers[0].input_data == NULL)
  {
    return -L2_ERROR;
  }

  network_layers[0].param_data = weights;
  if(network_layers[0].param_data == NULL)
  {
    return -L2_ERROR;
  }

  network_layers[0].output_data = pi_l2_malloc((unsigned int)(network_layers[0].layer_dim.x_out * network_layers[0].layer_dim.y_out * network_layers[0].layer_dim.c_out * sizeof(unsigned char)));
  if(network_layers[0].output_data == NULL)
  {
    return -L2_ERROR;
  }

#ifdef DEBUG
  printf("-> Exiting Layer Initialization...\n");
#endif

  return 0;
}

/*
 * Cluster application entry point
 */
void layer_run()
{
#ifdef DEBUG
  printf("--> Entering Layer Running...\n");
#endif

  pi_cl_dma_cmd_t dma_cmd;

  int dma_copy_size = l1_layer.layer_dim.x_ker * l1_layer.layer_dim.y_ker * l1_layer.layer_dim.c_in * l1_layer.layer_dim.c_out;

  /*
   * Parameters copy: Let's keep it simple, avoid the tiling along channels. Move all the parameters in L1
   */
#ifdef USE_L1_MEM
  pi_cl_dma_cmd((unsigned int)network_layers[0].param_data, (unsigned int)l1_layer.param_data, dma_copy_size, PI_CL_DMA_DIR_EXT2LOC, &dma_cmd);
  pi_cl_dma_cmd_wait(&dma_cmd);
#else
  l1_layer.param_data = network_layers[0].param_data;
#endif

#ifndef USE_L1_MEM
  l1_layer.output_data = network_layers[0].output_data;
#endif
  /*
   * Tile loop bounds
   */
  int nb_h_tile = 0;
  int nb_w_tile = 0;

  /*
   * Tile loop indexes
   */
  int h_tile = 0;
  int w_tile = 0;

  /*
   * Cluster performance counters
   */
#ifdef PERFORMANCE
  pi_perf_conf(1 << PI_PERF_ACTIVE_CYCLES | 1 << PI_PERF_CYCLES | 1 << PI_PERF_INSTR);
  pi_perf_reset();
  pi_perf_start();
#endif

#ifdef ENABLE_TILING
  nb_h_tile = network_layers[0].layer_dim.y_out / l1_layer.layer_dim.y_out;
  nb_w_tile = network_layers[0].layer_dim.x_out / l1_layer.layer_dim.x_out;
#else
  nb_h_tile = 1;
  nb_w_tile = 1;
#endif 


  /** EXERCISE4.1: implement double buffering
   *
   *  Double buffering pseudo code:
   *
   *    load tiles[0] -> current_buffer
   *    for i in range(len(tiles)):
   *        wait for previous loads and stores
   *        load tiles[i+1] -> next_buffer
   *        kernel_run current_buffer
   *        store current_buffer -> tiles[i]
   *    wait for previous loads and stores before exiting
   * 
   * To implement this, use functions from kernel_execute.h.
   * 
   * Summary of the functions:
   *  - void tile_load(int h_tile_idx, int w_tile_idx, int c_tile_idx, int buffer_idx)  -->  starts DMA transfer (L2->L1) for a specific tile (you can select which one with the indexes)
   *  - void kernel_run(int buffer_idx)                                                 -->  Runs inference for the selected buffer_idx
   *  - void tile_store(int h_tile_idx, int w_tile_idx, int c_tile_idx, int buffer_idx) -->  DMA copies the output from L1 to L2 memory
   *  - void tile_load_store_wait()                                                     -->  waits until load and store are complete
   */
  const int nb_tiles = nb_h_tile * nb_w_tile;
  int buffer_idx = 0;

  /*
  * Load tile 0's input data (outside the for loop)
  */
  /* YOUR CODE HERE */;

  for (int h_tile_idx = 0; h_tile_idx < nb_h_tile; h_tile_idx++) {
    for (int w_tile_idx = 0; w_tile_idx < nb_w_tile; w_tile_idx++) {
      const int next_buffer_idx = (buffer_idx + 1) % 2;
      const int next_w_tile_idx = (w_tile_idx + 1) % nb_w_tile;
      const int next_h_tile_idx = h_tile_idx + (next_w_tile_idx == 0 ? 1 : 0);  // If the next_w_tile is 0, that means we went into a new row, so the next_h_tile is h_tile + 1
      const int next_tile_idx = next_h_tile_idx * nb_w_tile + next_w_tile_idx;

      /*
       * Wait for the current tiles data
       * Note: we are waiting here and not right before the kernel, because
       * otherwise we would wait for the next tile too and ruin the point
       * of double buffering.
       */
      /* YOUR CODE HERE */;

      /*
      * Load next tiles input data from L2 into L1
      */
      if (next_tile_idx < nb_tiles) { // Check if there exists a 'next' tile
        /* YOUR CODE HERE */;
      }

      /*
       * Kernel execution
       */
      /* YOUR CODE HERE */;

      /*
       * Store output data back into L2 from L1
       */
      /* YOUR CODE HERE */;

      buffer_idx = next_buffer_idx;
    }
  }

  /*
  * Last wait for the DMA transfers
  */  
  /* YOUR CODE HERE */;

#ifdef PERFORMANCE
  pi_perf_stop();
  uint32_t instr_cnt      = pi_perf_read(PI_PERF_INSTR);
  uint32_t cycles_cnt     = pi_perf_read(PI_PERF_CYCLES);
  uint32_t act_cycles_cnt = pi_perf_read(PI_PERF_ACTIVE_CYCLES);
  printf("[0]: instructions = %d, tot_cycles = %d, active_cycles = %d \n", instr_cnt, cycles_cnt, act_cycles_cnt);
#endif

#ifdef DEBUG
  printf("--> Exiting Layer Running...\n");
#endif
}

/*
 * Check if the outputs are the same of the golden model
 */
int layer_check()
{
  int tot_layer_out_dim = network_layers[0].layer_dim.x_out * network_layers[0].layer_dim.y_out * network_layers[0].layer_dim.c_out;

  int errors = 0;
  for(int i=0; i<tot_layer_out_dim; i++)
  {
    if(output[i] != network_layers[0].output_data[i])
    {
      printf("ERROR at index %d, expected %x and got %x\n", i, output[i], network_layers[0].output_data[i]);
      errors++;
    }
  }

  if (errors != 0) {
    printf("Received %d errors\n", errors);
    return -GENERAL_ERROR;
  }

  printf("Exiting layer with 0 errors\n");

  return 0;
}

int layer_free()
{
#ifdef USE_L1_MEM
  pi_l1_free(&cluster, l1_layer.input_data , (unsigned int)(l1_layer.layer_dim.x_in  * l1_layer.layer_dim.y_in  * l1_layer.layer_dim.c_in * sizeof(unsigned char)));
  pi_l1_free(&cluster, l1_layer.param_data , (unsigned int)(l1_layer.layer_dim.x_ker * l1_layer.layer_dim.y_ker * l1_layer.layer_dim.c_in * l1_layer.layer_dim.c_out * sizeof(signed char)));
  pi_l1_free(&cluster, l1_layer.output_data, (unsigned int)(l1_layer.layer_dim.x_out * l1_layer.layer_dim.y_out * l1_layer.layer_dim.c_out * sizeof(unsigned char)));
#endif
  pi_cluster_close(&cluster);
  return 0;
}
