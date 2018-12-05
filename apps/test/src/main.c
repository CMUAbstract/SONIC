#include <msp430.h>
#include <stdlib.h>
#include <string.h>

#include <libio/console.h>
#include <libmspbuiltins/builtins.h>
#include <libmsp/mem.h>
#include <libmsp/periph.h>
#include <libmsp/clock.h>
#include <libmsp/watchdog.h>
#include <libmsp/gpio.h>

#include <libalpaca/alpaca.h>
#include <libfixed/fixed.h>
#include <libmat/mat.h>

#include <libdnn/misc.h>
#include <libdnn/mem.h>
#include <libdnn/state.h>
#include <libdnn/buffer.h>
#include <libdnn/nn.h>
#include <libdnn/nonlinear.h>
#include <libdnn/linalg.h>
#include <libdnn/profile.h>

#include "main.h"

#include "headers/a_dense.h"
#include "headers/a_sparse.h"
#include "headers/b_dense.h"
#include "headers/b_sparse.h"
#include "headers/c_dense.h"
#include "headers/c_sparse.h"


////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////Alapaca Shim///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#define MEM_SIZE 0x400
__hifram uint8_t *data_src[MEM_SIZE];
__hifram uint8_t *data_dest[MEM_SIZE];
__hifram unsigned int data_size[MEM_SIZE];
void clear_isDirty() {}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////Tasks///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void task_debug();
TASK(1, task_init);
TASK(2, task_compute);
TASK(3, task_exit);
TASK(4, task_debug);

ENTRY_TASK(task_init)
INIT_FUNC(init)

static void init_hw() {
	msp_watchdog_disable();
	msp_gpio_unlock();
	msp_clock_setup();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////Setup///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void init() {
	init_hw();

#ifdef CONFIG_CONSOLE
	#pragma message "init console"
	INIT_CONSOLE();
#endif

	__enable_interrupt();

	PRINTF(".%u.\r\n", curctx->task->idx);

    P2DIR = 0x00;   
    P3DIR = 0x00;   
    P4DIR = 0x00;   
    P7DIR = 0x00;   
    P8DIR = 0x00;   

    P6OUT = 0x00;
    P6DIR = 0x07;

    P5OUT = 0x01;
    P5DIR = 0x01;

#ifdef CONFIG_LED_DEBUG
	P1DIR = 0x01;
#else
	P1OUT = 0x00;
	P1DIR = 0x00;
#endif
}

////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////Stacks///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
__fram stack_t st;
__fram stack_t *mat_stack = &st;

////////////////////////////////////////////////////////////////////////////////
///////////////////////////////Weights Matrices/////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
__ro_fram mat_t mat_a_dense = {
	.dims = {15, 10},
	.strides = {10, 1},
	.len_dims = 2,
	.data = a_dense,
};

__ro_fram mat_t mat_a_sparse = {
	.dims = {A_SPARSE_LEN},
	.strides = {1},
	.len_dims = 1,
	.data = a_sparse,
	.sparse = {
		.dims = {15, 10},
		.len_dims = 2,
		.offsets = a_sparse_offsets,
		.sizes = a_sparse_sizes
	}
};

__ro_fram mat_t mat_b_dense = {
	.dims = {10, 1},
	.strides = {1, 1},
	.len_dims = 2,
	.data = b_dense,
};

__ro_fram mat_t mat_b_sparse = {
	.dims = {B_SPARSE_LEN},
	.strides = {1},
	.len_dims = 1,
	.data = b_sparse,
	.sparse = {
		.dims = {10, 1},
		.len_dims = 2,
		.offsets = b_sparse_offsets,
		.sizes = b_sparse_sizes
	}
};

__ro_fram mat_t mat_c_dense = {
	.dims = {3, 1, 3, 3},
	.len_dims = 4,
	.strides = {9, 9, 3, 1},
	.data = c_dense,
};

__ro_fram mat_t mat_c_sparse = {
	.dims = {C_SPARSE_LEN},
	.len_dims = 1,
	.strides = {1},
	.data = c_sparse,
	.sparse = {
		.dims = {3, 1, 3, 3},
		.len_dims = 4,
		.sizes = c_sparse_sizes,
		.offsets = c_sparse_offsets
	}
};

__fram mat_t buf1 = {.data = LAYER_BUFFER(1)};
__fram mat_t buf2 = {.data = LAYER_BUFFER(2)};
__fram mat_t *b1 = &buf1;
__fram mat_t *b2 = &buf2;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////Debug///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
__known fixed debug[0x100];
void task_debug() {
	MAT_DEBUG_DUMP(b1, 0, debug);
	TRANSITION_TO(task_exit);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////Tasks///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void task_init() {
	PRINTF("\r\n========================");
	PRINTF("\r\nInit");
#ifdef CONFIG_LED_DEBUG
	P1OUT = 0x01;
#endif
	TRANSITION_TO(task_compute);
}

void task_compute() {
	uint16_t state = CUR_SCRATCH[0];
	if(state == 0) {
		MAT_RESHAPE(b2, 10, 1);
		mat_t *mat_input_ptr = &mat_b_dense;
		for(uint16_t i = CUR_SCRATCH[1]; i < 10; i = ++CUR_SCRATCH[1]) {
			MAT_SET(b2, MAT_GET(mat_input_ptr, i, 0), i, 0);
			CUR_SCRATCH[2] = 0;
		}
		scratch_bak[0] = 1;
		scratch_bak[1] = 0;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		transition_to(CUR_TASK);
	} else if(state == 1) {
		TASK_REF(task_d_fc)->info.return_task = TASK_REF(task_compute);
		MAT_RESHAPE(b1, 15, 1);
		mat_t *w_ptr = &mat_a_dense;
		mat_t *b_ptr = NULL;
		// Assumes b, w, dest, src in that order
		PUSH_STACK(mat_stack, b_ptr, w_ptr, b1, b2);
		scratch_bak[0] = 2;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TRANSITION_TO(task_d_fc);
	} else if(state == 2) {
		PRINTF("\r\n dm_mul");
		MAT_RESHAPE(b1, 1, 15, 1);
		MAT_DUMP(b1, 0);
		MAT_RESHAPE(b1, 15, 1);

		MAT_RESHAPE(b2, 10, 1);
		mat_t *mat_input_ptr = &mat_b_dense;
		for(uint16_t i = CUR_SCRATCH[1]; i < 10; i = ++CUR_SCRATCH[1]) {
			MAT_SET(b2, MAT_GET(mat_input_ptr, i, 0), i, 0);
			CUR_SCRATCH[2] = 0;
		}
		scratch_bak[0] = 3;
		scratch_bak[1] = 0;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		transition_to(CUR_TASK);
	} else if(state == 3) {
		TASK_REF(task_s_fc)->info.return_task = TASK_REF(task_compute);
		MAT_RESHAPE(b1, 15, 1);
		mat_t *w_ptr = &mat_a_sparse;
		mat_t *b_ptr = NULL;
		// Assumes b, w, dest, src in that order
		PUSH_STACK(mat_stack, b_ptr, w_ptr, b1, b2);
		scratch_bak[0] = 4;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TRANSITION_TO(task_s_fc);
	} else if(state == 4) {
		PRINTF("\r\n svm_mul");
		MAT_RESHAPE(b1, 1, 15, 1);
		MAT_DUMP(b1, 0);
		MAT_RESHAPE(b1, 15, 1);
		TRANSITION_TO(task_debug);

		MAT_RESHAPE(b2, 1, 15, 10);
		mat_t *mat_input_ptr = &mat_a_dense;
		for(uint16_t i = CUR_SCRATCH[1]; i < 15; i = ++CUR_SCRATCH[1]) {
			for(uint16_t j = CUR_SCRATCH[2]; j < 10; j = ++CUR_SCRATCH[2]) {
				MAT_SET(b2, MAT_GET(mat_input_ptr, i, j), 0, i, j);
			}
			CUR_SCRATCH[2] = 0;
		}
		scratch_bak[0] = 5;
		scratch_bak[1] = 0;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		transition_to(CUR_TASK);
	} else if(state == 5) {
		params.same_padding = false;
		params.stride[0] = 1;
		params.stride[1] = 1;
		params.stride[2] = 1;
		TASK_REF(task_d_conv)->info.return_task = TASK_REF(task_compute);
		MAT_RESHAPE(b1, 3, 13, 8);
		mat_t *w_ptr = &mat_c_dense;
		mat_t *b_ptr = NULL;
		// Assumes b, w, dest, src in that order
		PUSH_STACK(mat_stack, b_ptr, w_ptr, b1, b2);
		scratch_bak[0] = 6;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TRANSITION_TO(task_d_conv);
	} else if(state == 6) {
		PRINTF("\r\n dm_conv");
		MAT_DUMP(b1, 0);
		MAT_DUMP(b1, 1);

		MAT_RESHAPE(b2, 1, 15, 10);
		mat_t *mat_input_ptr = &mat_a_dense;
		for(uint16_t i = CUR_SCRATCH[1]; i < 15; i = ++CUR_SCRATCH[1]) {
			for(uint16_t j = CUR_SCRATCH[2]; j < 10; j = ++CUR_SCRATCH[2]) {
				MAT_SET(b2, MAT_GET(mat_input_ptr, i, j), 0, i, j);
			}
			CUR_SCRATCH[2] = 0;
		}
		scratch_bak[0] = 7;
		scratch_bak[1] = 0;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		transition_to(CUR_TASK);
	} else if(state == 7) {
		params.same_padding = true;
		params.stride[0] = 1;
		params.stride[1] = 1;
		params.stride[2] = 1;
		TASK_REF(task_d_conv)->info.return_task = TASK_REF(task_compute);
		MAT_RESHAPE(b1, 3, 15, 10);
		mat_t *w_ptr = &mat_c_dense;
		mat_t *b_ptr = NULL;
		// Assumes b, w, dest, src in that order
		PUSH_STACK(mat_stack, b_ptr, w_ptr, b1, b2);
		scratch_bak[0] = 8;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TRANSITION_TO(task_d_conv);
	} else if(state == 8) {
		PRINTF("\r\n dm_conv - same");
		MAT_DUMP(b1, 0);
		MAT_DUMP(b1, 1);

		MAT_RESHAPE(b2, 1, 15, 10);
		mat_t *mat_input_ptr = &mat_a_dense;
		for(uint16_t i = CUR_SCRATCH[1]; i < 15; i = ++CUR_SCRATCH[1]) {
			for(uint16_t j = CUR_SCRATCH[2]; j < 10; j = ++CUR_SCRATCH[2]) {
				MAT_SET(b2, MAT_GET(mat_input_ptr, i, j), 0, i, j);
			}
			CUR_SCRATCH[2] = 0;
		}
		scratch_bak[0] = 9;
		scratch_bak[1] = 0;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		transition_to(CUR_TASK);
	} else if(state == 9) {
		params.same_padding = false;
		params.stride[0] = 1;
		params.stride[1] = 2;
		params.stride[2] = 2;
		TASK_REF(task_d_conv)->info.return_task = TASK_REF(task_compute);
		MAT_RESHAPE(b1, 3, 7, 4);
		mat_t *w_ptr = &mat_c_dense;
		mat_t *b_ptr = NULL;
		// Assumes b, w, dest, src in that order
		PUSH_STACK(mat_stack, b_ptr, w_ptr, b1, b2);
		scratch_bak[0] = 10;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TRANSITION_TO(task_d_conv);
	} else if(state == 10) {
		PRINTF("\r\n dm_conv - stride");
		MAT_DUMP(b1, 0);
		MAT_DUMP(b1, 1);

		MAT_RESHAPE(b2, 1, 15, 10);
		mat_t *mat_input_ptr = &mat_a_dense;
		for(uint16_t i = CUR_SCRATCH[1]; i < 15; i = ++CUR_SCRATCH[1]) {
			for(uint16_t j = CUR_SCRATCH[2]; j < 10; j = ++CUR_SCRATCH[2]) {
				MAT_SET(b2, MAT_GET(mat_input_ptr, i, j), 0, i, j);
			}
			CUR_SCRATCH[2] = 0;
		}
		scratch_bak[0] = 11;
		scratch_bak[1] = 0;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		transition_to(CUR_TASK);
	} else if(state == 11) {
		params.same_padding = false;
		params.stride[0] = 1;
		params.stride[1] = 1;
		params.stride[2] = 1;
		TASK_REF(task_s_conv)->info.return_task = TASK_REF(task_compute);
		MAT_RESHAPE(b1, 3, 13, 8);
		mat_t *w_ptr = &mat_c_sparse;
		mat_t *b_ptr = NULL;
		// Assumes b, w, dest, src in that order
		PUSH_STACK(mat_stack, b_ptr, w_ptr, b1, b2);
		scratch_bak[0] = 12;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TRANSITION_TO(task_s_conv);
	} else if(state == 12) {
		PRINTF("\r\n sm_conv");
		MAT_DUMP(b1, 0);
		MAT_DUMP(b1, 1);

		MAT_RESHAPE(b2, 1, 15, 10);
		mat_t *mat_input_ptr = &mat_a_dense;
		for(uint16_t i = CUR_SCRATCH[1]; i < 15; i = ++CUR_SCRATCH[1]) {
			for(uint16_t j = CUR_SCRATCH[2]; j < 10; j = ++CUR_SCRATCH[2]) {
				MAT_SET(b2, MAT_GET(mat_input_ptr, i, j), 0, i, j);
			}
			CUR_SCRATCH[2] = 0;
		}
		scratch_bak[0] = 13;
		scratch_bak[1] = 0;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		transition_to(CUR_TASK);
	} else if(state == 13) {
		params.same_padding = true;
		params.stride[0] = 1;
		params.stride[1] = 1;
		params.stride[2] = 1;
		TASK_REF(task_s_conv)->info.return_task = TASK_REF(task_compute);
		MAT_RESHAPE(b1, 3, 15, 10);
		mat_t *w_ptr = &mat_c_sparse;
		mat_t *b_ptr = NULL;
		// Assumes b, w, dest, src in that order
		PUSH_STACK(mat_stack, b_ptr, w_ptr, b1, b2);
		scratch_bak[0] = 14;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TRANSITION_TO(task_s_conv);
	} else if(state == 14) {
		PRINTF("\r\n sm_conv - same");
		MAT_DUMP(b1, 0);
		MAT_DUMP(b1, 1);

		MAT_RESHAPE(b2, 1, 15, 10);
		mat_t *mat_input_ptr = &mat_a_dense;
		for(uint16_t i = CUR_SCRATCH[1]; i < 15; i = ++CUR_SCRATCH[1]) {
			for(uint16_t j = CUR_SCRATCH[2]; j < 10; j = ++CUR_SCRATCH[2]) {
				MAT_SET(b2, MAT_GET(mat_input_ptr, i, j), 0, i, j);
			}
			CUR_SCRATCH[2] = 0;
		}
		scratch_bak[0] = 15;
		scratch_bak[1] = 0;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		transition_to(CUR_TASK);
	} else if(state == 15) {
		params.same_padding = false;
		params.stride[0] = 1;
		params.stride[1] = 2;
		params.stride[2] = 2;
		TASK_REF(task_s_conv)->info.return_task = TASK_REF(task_compute);
		MAT_RESHAPE(b1, 3, 7, 4);
		mat_t *w_ptr = &mat_c_sparse;
		mat_t *b_ptr = NULL;
		// Assumes b, w, dest, src in that order
		PUSH_STACK(mat_stack, b_ptr, w_ptr, b1, b2);
		scratch_bak[0] = 16;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TRANSITION_TO(task_s_conv);
	} else if(state == 16) {
		PRINTF("\r\n sm_conv - stride");
		MAT_DUMP(b1, 0);
		MAT_DUMP(b1, 1);
		exit(0);
	}
	TRANSITION_TO(task_exit);
}

void task_exit() {
    P6OUT = 0x01; // Turn on
    __delay_cycles(0x400);
    P6OUT = 0x00; // Turn off
	P1OUT = 0x01;
	P1DIR = 0x01;
	exit(0);
}

