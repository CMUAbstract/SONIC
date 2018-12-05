#ifndef MAIN_H
#define MAIN_H

void init();
void task_debug();
void task_toggle_timer();
void task_init();
void task_compute();
void task_finish();
void task_next_trial();
void task_exit();

#define DEBUG_AREA_SIZE 0x100

#ifndef CONFIG_CONSOLE
	#pragma message "no console"
	#define printf(fmt, ...) (void)0
#endif

#endif