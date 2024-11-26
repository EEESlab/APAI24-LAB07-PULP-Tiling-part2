#ifndef __KERNEL_H__
#define __KERNEL_H__


/*
 * Execute the convolution kernel
 */
void kernel_run(int buffer_idx);

/*
 * Load input tile data into L1
 *   - h_tile_idx, w_tile_idx, c_tile_idx - tile indexes used for L2 address calculation
 *   - buffer_idx - index of the buffer where to store the data in L1. In double buffering can be only 0 or 1.
 */
void tile_load(int  h_tile_idx, int  w_tile_idx, int  c_tile_idx, int buffer_idx);

/*
 * Store output tile data back to L2
 *   - h_tile_idx, w_tile_idx, c_tile_idx - tile indexes used for L2 address calculation
 *   - buffer_idx - index of the buffer where to load the data from L1. In double buffering can be only 0 or 1.
 */
void tile_store(int h_tile_idx, int w_tile_idx, int c_tile_idx, int buffer_idx);

/*
 * Wait for all previous loads and stores
 */
void tile_load_store_wait();


#endif // __KERNEL_H__
