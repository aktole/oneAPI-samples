// Header file to accompany board_test
#include <CL/sycl.hpp>
using namespace sycl;

constexpr size_t kKB = 1024;
constexpr size_t kMB = 1024 * 1024;
constexpr size_t kGB = 1024 * 1024 * 1024;

//////////////////////////////////
// **** PrintHelp function **** //
//////////////////////////////////

// Input:
// int details - Selection between long help or short help
// Returns:
// None

// The function does the following task:
// Prints short help with usage infomation and a longer help with details about
// each test

void PrintHelp(int details) {
  if (!details) {
    std::cout << "\n*** Board_test usage information ***\n"
              << "Command to run board_test using generated binary:\n"
              << "  > To run all tests (default): ./board_test.fpga\n"
              << "  > To run a specific test (see list below); pass the test "
              << "number as argument to \"-test\" option: ./board_test.fpga "
              << "-test=<test_number>\n"
              << "  > To see more details on what each test does: "
              << "./board_test.fpga -help\n"
              << "The tests are:\n"
              << "  1. Host Speed and Host Read Write Test\n"
              << "  2. Kernel Clock Frequency Test\n"
              << "  3. Kernel Launch Test\n"
              << "  4. Kernel Latency Measurement\n"
              << "  5. Kernel-to-Memory Read Write Test\n"
              << "  6. Kernel-to-Memory Bandwidth Test\n"
              << "Note: Kernel Clock Frequency is run along with all tests "
              << "except 1 (Host Speed and Host Read Write test)\n\n";
  } else {
    std::cout
        << "*** Board_test test details ***\n"
        << "Command to run board_test using generated binary:\n"
        << "  > To run all tests (default): ./board_test.fpga\n"
        << "  > To run a specific test (see list below); pass the test number "
        << "as argument to \"-test\" option: ./board_test.fpga "
        << "-test=<test_number>\n"
        << "The tests are:\n\n"
        << "  * 1. Host Speed and Host Read Write Test *\n"
        << "    Host Speed and Host Read Write test check the host to device "
        << "interface\n"
        << "    Host Speed test measures the host to device global memory "
        << "read, write as well as read-write bandwidth and reports it\n"
        << "    Host Read Write Test writes does unaligned memory to unaligned "
        << "device memory writes as well as reads from device to unaligned "
        << "host memory\n\n"
        << "  * 2. Kernel Clock Frequency Test *\n"
        << "    Kernel Clock Frequency Test measures the kernel clock "
        << "frequency of the bitstream running on the FPGA and compares this "
        << "to the Quartus compiled frequency for the kernel. This comparison "
        << "is turned on by default, see instructions below on how to alter "
        << "this default behavior\n"
        << "    This test expects the reports folder generated during compile "
        << "to be in one of the following locations : \n"
        << "      1. Same directory as the board_test.fpga binary\n"
        << "      2. Inside board_test.prj folder in the same directory as the "
        << "board_test.fpga binary\n"
        << "    If the reports folder is not found, the test will return only "
        << "the measured kernel frequency and fail, none of the other tests "
        << "will run as hardware frequency may not be the expected value and "
        << "may lead to errors.\n"
        << "    If you wish to override this failure, please set "
        << "\"report_chk\" variable to \"false\" in <board_test.cpp> and "
        << "recompile host code only.\n\n"
        << "  * 3. Kernel Launch Test *\n"
        << "    Kernel Launch test checks if the kernel launched and executed "
        << "successfully. This is done by launching a sender kernel that "
        << "writes a value to a pipe, this pipe is read by the receiver "
        << "kernel, which completes if the correct value is read.\n"
        << "    This test will hang if the receiver kernel does not receive "
        << "the correct value\n\n"
        << "  * 4. Kernel Latency Measurement *\n"
        << "    This test measures the round trip kernel latency by launching "
        << "a no-operation kernel\n\n"
        << "  * 5. Kernel-to-Memory Read Write Test *\n"
        << "    Kernel-to-Memory Read Write test checks kernel to device "
        << "global memory interface. The test writes data to the entire device "
        << "global memory from host; the kernel then reads -> modifies and "
        << "writes the data back to the device global memory."
        << "    The host reads the modified data back and verifies the read "
        << "back values match expected value\n"
        << "  * 6. Kernel-to-Memory Bandwidth Test *\n"
        << "    Kernel-to-Memory Bandwidth test measures the kernel to device "
        << "global memory bandwidth and compares this with the theoretical "
        << "bandwidth defined in board_spec.xml file in the oneAPI shim "
        << "provided the <HLD_SHIM_ROOT_HW> environment variable is set to "
        << "point to <path-to-oneAPI-shim/hardware/<board_variant>/> "
        << "directory.\n"
        << "    If this environment variable is not set, the test only reports "
        << "the measured kernel to memory bandwidth,\n\n"
        << "    Note: This test assumes that design was compiled with "
        << "-Xsno-interleaving option\n\n"
        << "Please use the commands shown at the beginning of this help to run "
        << "all or one of the above tests\n\n";
  }
}  // End of PrintHelp

//////////////////////////////////////////////
// **** SyclGetQSubExecTimeNs function **** //
//////////////////////////////////////////////

// Input:
// event e - Sycl event with profiling information
// Returns:
// Difference in time from command submission to command end (in nanoseconds)

// The function does the following task:
// Gets profiling information from a Sycl event and
// returns execution time for a given SYCL event from a queue

unsigned long SyclGetQSubExecTimeNs(event e) {
  unsigned long submit_time =
      e.get_profiling_info<info::event_profiling::command_submit>();
  unsigned long end_time =
      e.get_profiling_info<info::event_profiling::command_end>();
  return (end_time - submit_time);
}  // End of SyclGetQSubExecTimeNs

/////////////////////////////////////////////
// **** SyclGetQStExecTimeNs function **** //
/////////////////////////////////////////////

// Input:
// event e - Sycl event with profiling information
// Returns:
// Difference in time from command start to command end (in nanoseconds)

// The function does the following task:
// Gets profiling information from a Sycl event and
// returns execution time for a given SYCL event from a queue

unsigned long SyclGetQStExecTimeNs(event e) {
  unsigned long start_time =
      e.get_profiling_info<info::event_profiling::command_start>();
  unsigned long end_time =
      e.get_profiling_info<info::event_profiling::command_end>();
  return (end_time - start_time);
}  // End of SyclGetQStExecTimeNs

///////////////////////////////////////////
// **** SyclGetTotalTimeNs function **** //
///////////////////////////////////////////

// Input:
// event first_evt - Sycl event with profiling information
// event last_evt - another Sycl event with profiling information
// Returns:
// Difference in time from command submission of first event to command end of
// last event (in nanoseconds)

// The function does the following task:
// Gets profiling information from two different Sycl events and
// returns the total execution time for all events between first and last

unsigned long SyclGetTotalTimeNs(event first_evt, event last_evt) {
  unsigned long first_evt_submit =
      first_evt.get_profiling_info<info::event_profiling::command_submit>();
  unsigned long last_evt_end =
      last_evt.get_profiling_info<info::event_profiling::command_end>();
  return (last_evt_end - first_evt_submit);
}  // End of SyclGetTotalTimeNs

/////////////////////////////////////////
// **** InitializeVector function **** //
/////////////////////////////////////////

// Inputs:
// 1. unsigned *vector - pointer to host memory that has to be initialized
// (allocated in calling function)
// 2. size_t size - number of elements to initialize
// 3. size_t offset - value to use for initialization
// Returns:
// None

// The function does the following task:
// Initializes "size" number of elements in memory pointed
// to with "offset + i", where i is incremented by loop controlled by "size"

void InitializeVector(unsigned *vector, size_t size, size_t offset) {
  for (size_t i = 0; i < size; ++i) {
    vector[i] = offset + i;
  }
}

/////////////////////////////////////////
// **** InitializeVector function **** //
/////////////////////////////////////////

// Inputs:
// 1. unsigned *vector - pointer to host memory that has to be initialized
// (allocated in calling function)
// 2. size_t size - number of elements to initialize
// Returns:
// None

// The function does the following task:
// Initializes "size" number of elements in memory pointed
// to with random values (output of rand() function)

void InitializeVector(unsigned *vector, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    vector[i] = rand();
  }
}