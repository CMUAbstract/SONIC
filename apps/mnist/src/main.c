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
#ifndef CONFIG_DENSE
#include "headers/conv1.h"
#else
#include "headers/dense/conv1.h"
#endif
#include "headers/conv2.h"
#include "headers/fc1.h"
#include "headers/fc2.h"
#include "headers/input.h"


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
TASK(0, task_toggle_timer);
TASK(1, task_init);
TASK(2, task_compute);
TASK(3, task_finish);
TASK(4, task_next_trial);
TASK(5, task_exit);
TASK(6, task_debug);

ENTRY_TASK(task_init)
INIT_FUNC(init)

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////Debug///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
__known fixed debug_area[DEBUG_AREA_SIZE];
__fram uint16_t timer_bak = 0;
void task_toggle_timer() {
	if(CUR_SCRATCH[1] == 0) { // Make sure we die, to start again with full tank
		CUR_SCRATCH[1] = 1;
#ifdef CONFIG_INTERMITTENT
		P1OUT = 0x01;
		P1DIR = 0x01;
		while(1) {}
#endif
	}
	uint16_t toggle = CUR_SCRATCH[0];
	P6OUT = toggle; // Turn on
	__delay_cycles(0x400);
	P6OUT = 0x00; // Turn off
	debug_area[3]++;
	write_to_gbuf((uint8_t *)(&timer_bak), 
		(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	transition_to(CUR_INFO.return_task);
}

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
	debug_area[0] = curctx->task->idx;
	debug_area[1]++;

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
__ro_fram mat_t mat_conv1_wd = {
#ifndef CONFIG_DENSE
	.dims = {CONV1_WD_LEN},
	.len_dims = 1,
	.strides = {1},
	.data = conv1_wd,
	.sparse = {
		.dims = {20, 1, 1, 1},
		.len_dims = 4,
		.sizes = conv1_wd_sizes,
		.offsets = conv1_wd_offsets
	},
#else
	.dims = {20, 1, 1, 1},
	.len_dims = 4,
	.strides = {1, 1, 1, 1},
	.data = conv1_wd,
#endif
};

__ro_fram mat_t mat_conv1_wv = {
#ifndef CONFIG_DENSE
	.dims = {CONV1_WV_LEN},
	.len_dims = 1,
	.strides = {1},
	.data = conv1_wv,
	.sparse = {
		.dims = {20, 1, 5, 1},
		.len_dims = 4,
		.sizes = conv1_wv_sizes,
		.offsets = conv1_wv_offsets
	},
#else
	.dims = {20, 1, 5, 1},
	.len_dims = 4,
	.strides = {5, 5, 5, 1},
	.data = conv1_wv,
#endif
};

__ro_fram mat_t mat_conv1_wh = {
#ifndef CONFIG_DENSE
	.dims = {CONV1_WH_LEN},
	.len_dims = 1,
	.strides = {1},
	.data = conv1_wh,
	.sparse = {
		.dims = {20, 1, 1, 5},
		.len_dims = 4,
		.sizes = conv1_wh_sizes,
		.offsets = conv1_wh_offsets
	},
#else
	.dims = {20, 1, 1, 5},
	.len_dims = 5,
	.strides = {5, 5, 5, 1},
	.data = conv1_wh,
#endif
};

__ro_fram mat_t mat_conv1_b = {
	.dims = {20},
	.len_dims = 1,
	.strides = {1},
	.data = conv1_b,
};

__ro_fram mat_t  mat_conv2_w = {
	.dims = {CONV2_W_LEN},
	.len_dims = 1,
	.strides = {1},
	.data = conv2_w,
	.sparse = {
		.dims = {100, 20, 5, 5},
		.len_dims = 4,
		.sizes = conv2_w_sizes,
		.offsets = conv2_w_offsets,
	}
};

__ro_fram mat_t mat_conv2_b = {
	.dims = {100},
	.strides = {1},
	.len_dims = 1,
	.data = conv2_b,
};

__ro_fram mat_t mat_fc1_wh = {
	.dims = {FC1_WH_LEN},
	.len_dims = 1,
	.strides = {1},
	.data = fc1_wh,
	.sparse = {
		.dims = {100, 1600},
		.len_dims = 2,
		.offsets = fc1_wh_offsets,
		.sizes = fc1_wh_sizes,
	},
};

__ro_fram mat_t mat_fc1_wv = {
	.dims = {FC1_WV_LEN},
	.len_dims = 1,
	.strides = {1},
	.data = fc1_wv,
	.sparse = {
		.dims = {500, 100},
		.len_dims = 2,
		.offsets = fc1_wv_offsets,
		.sizes = fc1_wv_sizes,
	},
};

__ro_fram mat_t mat_fc1_b = {
	.dims = {500, 1},
	.strides = {1, 1},
	.len_dims = 2,
	.data = fc1_b,
};

__ro_fram mat_t mat_fc2_w = {
	.dims = {10, 500},
	.strides = {500, 1},
	.len_dims = 2,
	.data = fc2_w,
};

__ro_fram mat_t mat_fc2_b = {
	.dims = {10, 1},
	.strides = {1, 1},
	.len_dims = 2,
	.data = fc2_b,
};

__ro_fram mat_t mat_input = {
	.dims = {1, 28, 28},
	.strides = {784, 28, 1},
	.len_dims = 3,
	.data = input,
};

__fram mat_t buf1 = {.data = LAYER_BUFFER(1)};
__fram mat_t buf2 = {.data = LAYER_BUFFER(2)};
__fram mat_t *b1 = &buf1;
__fram mat_t *b2 = &buf2;

void task_debug() {
	P1OUT = 0x00;
	MAT_DEBUG_DUMP(b1, 1, debug_area + 4);
	while(1) {}
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////Tasks///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
__fram uint16_t trials = 0;
__fram uint16_t start_experiment = 0;
void task_init() {
	PRINTF("\r\n========================");
	PRINTF("\r\nInit");
#ifdef CONFIG_LED_DEBUG
	P1OUT = 0x01;
#endif

#ifndef CONFIG_CONT
	if(trials == 0 && CUR_SCRATCH[0] == 0) {
		CUR_SCRATCH[0] = 1;
		P1OUT = 0x01;
		P1DIR = 0x01;
		while(1) {}
	}
#endif

	params.same_padding = false;
	params.size[0] = 1;
	params.size[1] = 2;
	params.size[2] = 2;
	params.stride[0] = 1;
	params.stride[1] = 1;
	params.stride[2] = 1;

	TASK_REF(task_toggle_timer)->info.return_task = TASK_REF(task_compute);
	TASK_REF(task_toggle_timer)->info.scratch[0] = 0x02;
	TRANSITION_TO(task_toggle_timer);
}

void task_compute() {
	uint16_t state = CUR_SCRATCH[0];
	debug_area[2] = state;
	if(state == 0) {
		MAT_RESHAPE(b2, 1, 28, 28);
		mat_t *mat_input_ptr = &mat_input;
		for(uint16_t i = CUR_SCRATCH[1]; i < 28; i = ++CUR_SCRATCH[1]) {
			for(uint16_t j = CUR_SCRATCH[2]; j < 28; j = ++CUR_SCRATCH[2]) {
				fixed w = MAT_GET(mat_input_ptr, 0, i, j);
				MAT_SET(b2, w, 0, i, j);
			}
			CUR_SCRATCH[2] = 0;
		}
		scratch_bak[0] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		transition_to(CUR_TASK);
	}
#ifndef CONFIG_DENSE 
	else if(state == 1) {
		MAT_DUMP(b2, 0);
		PRINTF("\r\n Layer 1");
		// TASK_REF(task_d_conv)->info.return_task = TASK_REF(task_compute);
		TASK_REF(task_s_conv)->info.return_task = TASK_REF(task_compute);
		MAT_RESHAPE(b1, 20, 28, 28);
		mat_t *w_ptr = &mat_conv1_wd;
		mat_t *b_ptr = NULL;
		// Assumes b, w, dest, src in that order
		PUSH_STACK(mat_stack, b_ptr, w_ptr, b1, b2);
		scratch_bak[0] = 2;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TASK_REF(task_toggle_timer)->info.return_task = TASK_REF(task_s_conv);
		TASK_REF(task_toggle_timer)->info.scratch[0] = 0x04;
		TRANSITION_TO(task_toggle_timer);
	} else if(state == 2) {
		MAT_DUMP(b1, 0);
		PRINTF("\r\n Layer 2");
		TASK_REF(task_s_depthconv)->info.return_task = TASK_REF(task_compute);
		MAT_RESHAPE(b2, 20, 28, 24);
		mat_t *w_ptr = &mat_conv1_wv;
		mat_t *b_ptr = NULL;
		// Assumes b, w, dest, src in that order
		PUSH_STACK(mat_stack, b_ptr, w_ptr, b2, b1);
		scratch_bak[0] = 3;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TRANSITION_TO(task_s_depthconv);
	} else if(state == 3) {
		MAT_DUMP(b2, 0);
		PRINTF("\r\n Layer 3");
		// TASK_REF(task_d_conv)->info.return_task = TASK_REF(task_compute);
		TASK_REF(task_s_depthconv)->info.return_task = TASK_REF(task_compute);
		MAT_RESHAPE(b1, 20, 24, 24);
		mat_t *w_ptr = &mat_conv1_wh;
		mat_t *b_ptr = &mat_conv1_b;
		// Assumes b, w, dest, src in that order
		PUSH_STACK(mat_stack, b_ptr, w_ptr, b1, b2);
		scratch_bak[0] = 4;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TRANSITION_TO(task_s_depthconv);
	} 
#else
	else if(state == 1) {
		MAT_DUMP(b2, 0);
		PRINTF("\r\n Layer 1");
		// TASK_REF(task_d_conv)->info.return_task = TASK_REF(task_compute);
		TASK_REF(task_d_conv)->info.return_task = TASK_REF(task_compute);
		MAT_RESHAPE(b1, 20, 28, 28);
		mat_t *w_ptr = &mat_conv1_wd;
		mat_t *b_ptr = &mat_conv1_b;
		// Assumes b, w, dest, src in that order
		PUSH_STACK(mat_stack, b_ptr, w_ptr, b1, b2);
		scratch_bak[0] = 2;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TASK_REF(task_toggle_timer)->info.return_task = TASK_REF(task_d_conv);
		TASK_REF(task_toggle_timer)->info.scratch[0] = 0x04;
		TRANSITION_TO(task_toggle_timer);
	} else if(state == 2) {
		MAT_DUMP(b1, 0);
		PRINTF("\r\n Layer 2");
		TASK_REF(task_d_depthconv)->info.return_task = TASK_REF(task_compute);
		MAT_RESHAPE(b2, 20, 28, 24);
		mat_t *w_ptr = &mat_conv1_wv;
		mat_t *b_ptr = &mat_conv1_b;
		// Assumes b, w, dest, src in that order
		PUSH_STACK(mat_stack, b_ptr, w_ptr, b2, b1);
		scratch_bak[0] = 3;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TRANSITION_TO(task_d_depthconv);
	} else if(state == 3) {
		MAT_DUMP(b2, 0);
		PRINTF("\r\n Layer 3");
		// TASK_REF(task_d_conv)->info.return_task = TASK_REF(task_compute);
		TASK_REF(task_d_depthconv)->info.return_task = TASK_REF(task_compute);
		MAT_RESHAPE(b1, 20, 24, 24);
		mat_t *w_ptr = &mat_conv1_wh;
		mat_t *b_ptr = &mat_conv1_b;
		// Assumes b, w, dest, src in that order
		PUSH_STACK(mat_stack, b_ptr, w_ptr, b1, b2);
		scratch_bak[0] = 4;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TRANSITION_TO(task_d_depthconv);
	} 
#endif
	else if(state == 4) {
		prof(SECTION, "conv1");
		MAT_DUMP(b1, 1);
		PRINTF("\r\n Layer 4");
		TASK_REF(task_relu)->info.return_task = TASK_REF(task_compute);
		MAT_RESHAPE(b2, 20, 24, 24);
		// Assumes filter, dest, src in that order
		PUSH_STACK(mat_stack, b2, b1);
		scratch_bak[0] = 5;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TASK_REF(task_toggle_timer)->info.return_task = TASK_REF(task_relu);
		TASK_REF(task_toggle_timer)->info.scratch[0] = 0x04;
		TRANSITION_TO(task_toggle_timer);
		// TRANSITION_TO(task_relu);
	} else if(state == 5) {
		MAT_DUMP(b2, 1);
		PRINTF("\r\n Layer 5");
		TASK_REF(task_pool)->info.return_task = TASK_REF(task_compute);
		MAT_RESHAPE(b1, 20, 12, 12);
		params.stride[1] = 2;
		params.stride[2] = 2;
		// Assumes filter, dest, src in that order
		PUSH_STACK(mat_stack, b1, b2);
		scratch_bak[0] = 6;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TRANSITION_TO(task_pool);
	} else if(state == 6) {
		prof(SECTION, "place1");
		MAT_DUMP(b1, 1);
		PRINTF("\r\n Layer 6");
		TASK_REF(task_s_conv)->info.return_task = TASK_REF(task_compute);
		MAT_RESHAPE(b2, 100, 12, 12);
		params.stride[1] = 1;
		params.stride[2] = 1;
		mat_t *w_ptr = &mat_conv2_w;
		mat_t *b_ptr = &mat_conv2_b;
		// Assumes b, w, dest, src in that order
		PUSH_STACK(mat_stack, b_ptr, w_ptr, b2, b1);
		scratch_bak[0] = 7;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TASK_REF(task_toggle_timer)->info.return_task = TASK_REF(task_s_conv);
		TASK_REF(task_toggle_timer)->info.scratch[0] = 0x04;
		TRANSITION_TO(task_toggle_timer);
		// TRANSITION_TO(task_s_conv);
	} else if(state == 7) {
		prof(SECTION, "conv2");
		MAT_DUMP(b2, 1);
		PRINTF("\r\n Layer 7");
		TASK_REF(task_relu)->info.return_task = TASK_REF(task_compute);
		MAT_RESHAPE(b1, 100, 8, 8);
		// Assumes filter, dest, src in that order
		PUSH_STACK(mat_stack, b1, b2);
		scratch_bak[0] = 8;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TRANSITION_TO(task_relu);
	} else if(state == 8) {
		MAT_DUMP(b1, 10);
		PRINTF("\r\n Layer 8");
		TASK_REF(task_pool)->info.return_task = TASK_REF(task_compute);
		MAT_RESHAPE(b2, 100, 4, 4);
		params.stride[1] = 2;
		params.stride[2] = 2;
		// Assumes filter, dest, src in that order
		PUSH_STACK(mat_stack, b2, b1);
		scratch_bak[0] = 9;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TASK_REF(task_toggle_timer)->info.return_task = TASK_REF(task_pool);
		TASK_REF(task_toggle_timer)->info.scratch[0] = 0x04;
		TRANSITION_TO(task_toggle_timer);
	} else if(state == 9) {
		prof(SECTION, "place2");
		MAT_RESHAPE(b2, 1, 1, 1600);
		MAT_DUMP(b2, 0);
		PRINTF("\r\n Layer 9");
		TASK_REF(task_s_fc)->info.return_task = TASK_REF(task_compute);
		MAT_RESHAPE(b2, 1600, 1);
		MAT_RESHAPE(b1, 100, 1);
		mat_t *w_ptr = &mat_fc1_wh;
		mat_t *b_ptr = NULL;
		// Assumes b, w, dest, src in that order
		PUSH_STACK(mat_stack, b_ptr, w_ptr, b1, b2);
		scratch_bak[0] = 10;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		// TRANSITION_TO(task_s_fc);
		TASK_REF(task_toggle_timer)->info.return_task = TASK_REF(task_s_fc);
		TASK_REF(task_toggle_timer)->info.scratch[0] = 0x04;
		TRANSITION_TO(task_toggle_timer);
	} else if(state == 10) {
		PRINTF("\r\n Layer 10");
		TASK_REF(task_s_fc)->info.return_task = TASK_REF(task_compute);
		MAT_RESHAPE(b2, 500, 1);
		mat_t *w_ptr = &mat_fc1_wv;
		mat_t *b_ptr = &mat_fc1_b;
		// Assumes b, w, dest, src in that order
		PUSH_STACK(mat_stack, b_ptr, w_ptr, b2, b1);
		scratch_bak[0] = 11;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TRANSITION_TO(task_s_fc);
	} else if(state == 11) {
		MAT_RESHAPE(b2, 1, 1, 500);
		MAT_DUMP(b2, 0);
		MAT_RESHAPE(b2, 500, 1);
		PRINTF("\r\n Layer 11");
		TASK_REF(task_relu)->info.return_task = TASK_REF(task_compute);
		MAT_RESHAPE(b1, 500, 1);
		// Assumes dest, src in that order
		PUSH_STACK(mat_stack, b1, b2);
		scratch_bak[0] = 12;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TRANSITION_TO(task_relu);
	} else if(state == 12) {
		MAT_RESHAPE(b1, 1, 1, 500);
		MAT_DUMP(b1, 0);
		MAT_RESHAPE(b1, 500, 1);
		PRINTF("\r\n Layer 12");
		TASK_REF(task_d_fc)->info.return_task = TASK_REF(task_compute);
		MAT_RESHAPE(b2, 10, 1);
		mat_t *w_ptr = &mat_fc2_w;
		mat_t *b_ptr = &mat_fc2_b;
		// Assumes b, w, dest, src in that order
		PUSH_STACK(mat_stack, b_ptr, w_ptr, b2, b1);
		scratch_bak[0] = 13;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TRANSITION_TO(task_d_fc);
	}
	prof(SECTION, "fc");
	// TRANSITION_TO(task_finish);
	TASK_REF(task_toggle_timer)->info.return_task = TASK_REF(task_finish);
	TASK_REF(task_toggle_timer)->info.scratch[0] = 0x04;
	TRANSITION_TO(task_toggle_timer);
}

void task_finish() {
	TASK_REF(task_toggle_timer)->info.return_task = TASK_REF(task_next_trial);
	TASK_REF(task_toggle_timer)->info.scratch[0] = 0x02;
	TRANSITION_TO(task_toggle_timer);
}

__fram fixed max = 0;
__fram uint16_t predict = 0;
void task_next_trial() {
	fixed max = 0;
	PRINTF("\r\n=====================");
	for(uint16_t i = CUR_SCRATCH[0]; i < 10; i = ++CUR_SCRATCH[0]) {
		fixed v = MAT_GET(b2, i, 0);
		if(v > max) {
			predict = i;
			max = v;
		}
		debug_area[i + 4] = v;
		PRINTF("\r\n%u => %i", i, v);
	}
	debug_area[3] = predict;
	prof_print();
	PRINTF("\r\n=====================");
	PRINTF("\r\n=====================");
	if(trials + 1 < CONFIG_TRIALS) {
		trials++;
		mat_stack->pos = 0;
		memset(TASK_REF(task_compute)->info.scratch, 0, sizeof(uint16_t) * SCRATCH_SIZE);
		CUR_SCRATCH[0] = 0;
		curctx->needCommit = 0;
		curctx->task = TASK_REF(task_init);
#ifndef CONFIG_CONT
		P1OUT = 0x01;
		P1DIR = 0x01;
		while(1) {}
#else
		TRANSITION_TO(task_init);
#endif
	}
	TASK_REF(task_toggle_timer)->info.return_task = TASK_REF(task_exit);
	TASK_REF(task_toggle_timer)->info.scratch[0] = 0x01;
	TRANSITION_TO(task_toggle_timer);
}

void task_exit() {
	P1OUT = 0x00;
	exit(0);
}

