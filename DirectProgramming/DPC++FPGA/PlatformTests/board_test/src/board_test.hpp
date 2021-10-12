#include <dirent.h>
#include <limits.h>
#include <CL/sycl.hpp>
#include <vector>

#include "host_speed.hpp"

// Pre-declare kernel name to prevent name mangling
// This is an FPGA best practice that makes it easier to identify the kernel in
// the optimization reports.
class NopNDRange;
class NopSingleTask;
class KernelSender;
class KernelReceiver;
class MemReadWriteStream;
class MemReadWriteStreamNDRange;
class MemWriteStream;
class MemReadStream;

// Pipe used between KernelSender and KernelReceiver -
// ShimMetrics::KernelLaunchTest(queue &q) function
using SendertoReceiverPipe = INTEL::pipe<  // Defined in the SYCL headers
    class SenderReceiverPipe,              // An identifier for the pipe
    unsigned int,                          // The type of data in the pipe
    1>;                                    // The capacity of the pipe

/////////////////////////////////
// **** class ShimMetrics **** //
/////////////////////////////////

// Object stores oneAPI/OpenCL shim metrics
// Member Functions (details closer to function definition):
// ShimMetrics - Constructor; initializes all metrics and obtains maximum device
// allocation and maxmum device global memory TestGlobalMem - Host to device
// global memory interface check HostSpeed - Host to device global memory
// bandwidth measurement HostRWTest - Unaligned read & writes from host to
// device global memory KernelClkFreq - Kernel clock frequency measurement
// KernelLatency - Kernel latency measurement
// KernelLaunchTest - Host to kernel interface check
// KernelMemRW - Kernel to device global memory interface check
// KernelMemBW - Kernel to device global memory bandwidth measurement

class ShimMetrics {
 public:
  ShimMetrics(queue &q)
      : h2d_rd_bw{0},
        h2d_wr_bw{0},
        h2d_rd_wr_bw{0},
        h2d_rw_test{false},
        kernel_freq{0},
        kernel_latency{0},
        kernel_thruput{0},
        kernel_mem_bw{0},
        kernel_mem_rw_test{false} {
    max_buffer_size = q.get_device().get_info<info::device::global_mem_size>();
    max_alloc_size =
        q.get_device().get_info<info::device::max_mem_alloc_size>();
    std::cout << "\nclGetDeviceInfo CL_DEVICE_GLOBAL_MEM_SIZE = "
              << max_buffer_size << "\n";
    std::cout << "clGetDeviceInfo CL_DEVICE_MAX_MEM_ALLOC_SIZE = "
              << max_alloc_size << "\n";
    std::cout << "Device buffer size available for allocation = "
              << max_alloc_size << " bytes\n";
  }

  ~ShimMetrics() {}

  size_t TestGlobalMem(queue &q);
  int HostSpeed(queue &q);
  int HostRWTest(queue &q, size_t dev_offset = 0);
  int KernelClkFreq(queue &q, bool report_chk = true);
  int KernelLatency(queue &q);
  int KernelLaunchTest(queue &q);
  int KernelMemRW(queue &q);
  int KernelMemBW(queue &q);

 private:
  float h2d_rd_bw;
  float h2d_wr_bw;
  float h2d_rd_wr_bw;
  bool h2d_rw_test;
  float kernel_freq;
  float kernel_latency;
  float kernel_thruput;
  float kernel_mem_bw;
  bool kernel_mem_rw_test;
  cl_ulong max_buffer_size;
  cl_ulong max_alloc_size;
};

//////////////////////////////////////
// **** TestGlobalMem function **** //
//////////////////////////////////////

// Input:
// queue &q - queue to submit operation
// Returns:
// Number of errors in transfer OR 1 if device memory allocation is 0 or test
// fails (return 0 means test passed)

// The function does the following tasks:
// 1. Get maximum device global memory allocation size
// 2. Allocate host memory to use to store data to be written to and read from
// device
// 3. Write to device global memory
// 4. Read from device global memory
// 5. Verify data read back matches value written
// 6. Report read, write bandwidth
// If this test passes (returns 0), the host to device global memory interface
// is working fine

size_t ShimMetrics::TestGlobalMem(queue &q) {
  // Data is transferred from host to device in kMaxHostChunk size transfers
  // (size in bytes)
  constexpr size_t kMaxHostChunk = 512 * kMB;

  // Test fails if max alloc size is 0
  if (max_alloc_size == 0) {
    std::cout << "Maximum global memory allocation supported by Sycl device is "
              << "0! Cannot run kernel-to-memory read wite test\n\n";
    return 1;
  }

  // **** Create device buffer ****//
  // Creating device buffer to span all usable global memory space on device
  buffer<unsigned long, 1> dev_buf{
      range<1>{(max_alloc_size / sizeof(unsigned long))}};
  std::cout << "Size of buffer created = " << dev_buf.get_size() << " bytes\n";

  // **** Host memory allocation **** //
  /*<hostbuf> is used to store data
  to be written to device buffer as well as data that is read back
  While loop retries allocation for smaller chunk if kMaxHostChunk allocation
  fails */
  // Size of allocated host memory (in bytes)
  size_t host_size = kMaxHostChunk;
  unsigned long *hostbuf = (unsigned long *)malloc(host_size);
  while (hostbuf == NULL) {
    host_size = host_size / 2;
    hostbuf = (unsigned long *)malloc(host_size);
  }

  // **** Writing data from host memory to device global memory **** //

  // Number of bytes remaining to transfer to device memory
  unsigned long bytes_rem = max_alloc_size;
  // offset at which write should begin in device global memory
  unsigned long offset = 0;
  // Number of chunks of host_size written to device
  unsigned long chunk_cnt_wr = 0;
  // Total time to write
  double sum_time_ns = 0.0;

  // Copying host memory to device buffer in chunks
  std::cout << "Writing " << (max_alloc_size / kMB)
            << " MB to device global memory ... ";

  // Device global memory larger (i.e. max_alloc_size) than host memory size
  // (i.e. host_size) Chunks of host memory size written to device memory in
  // iterations
  while (bytes_rem > 0) {
    unsigned long chunk = bytes_rem;
    if (chunk > host_size) chunk = host_size;

    // Initializing host buffer
    for (unsigned long i = 0; i < chunk / sizeof(unsigned long); ++i) {
      hostbuf[i] = offset + i;
    }

    // Submit command to copy (explicit copy from host to device)
    auto h2d_copy_e = q.submit([&](handler &h) {
      // Range of buffer that needs to accessed
      auto buf_range = chunk / sizeof(unsigned long);
      // offset starts at 0 - incremented by chunk size each iteration
      auto buf_offset = offset / sizeof(unsigned long);
      // Access host_size range of buffer starting at buf_offset
      accessor mem(dev_buf, h, buf_range, buf_offset);
      // Writing from host memory to device buffer
      h.copy(hostbuf, mem);
    });

    // Wait for explicit copy from host memory to device buffer to complete
    h2d_copy_e.wait();

    // Get time for copy operation using Sycl event information (return in
    // nanoseconds)
    sum_time_ns += SyclGetQSubExecTimeNs(h2d_copy_e);

    // Increment offset and decrement remaining bytes by size of transfer
    offset += chunk;
    bytes_rem -= chunk;
    chunk_cnt_wr++;

  }  // End of write-to-device while loop

  // Report write bandwidth
  std::cout << (((float)max_alloc_size / kMB) /
                ((float)sum_time_ns / 1000000000))
            << " MB/s\n";

  // **** Reading data from device global memory to host memory **** //

  // Read back all of memory and verify
  std::cout << "Reading " << (max_alloc_size / kMB)
            << " MB from device global memory ... ";

  // Reset variables for read loop
  bytes_rem = max_alloc_size;
  // Start reading at offset 0
  offset = 0;
  // Total time to read
  sum_time_ns = 0.0;

  // The same host memory is used to read back values, resetting it to 0
  for (unsigned long i = 0; i < host_size / sizeof(unsigned long); ++i) {
    hostbuf[i] = 0;
  }

  // Variables for error calculation (verify value read back matches written
  // value)
  unsigned long errors = 0;
  unsigned long compare_count = 0;
  unsigned long chunk_errors = 0;
  unsigned long chunk_cnt_rd = 0;

  // Device global memory larger (i.e. max_alloc_size) than host memory size
  // (i.e. host_size) Read back chunks of host memory size from device memory in
  // iterations
  while (bytes_rem > 0) {
    unsigned long chunk = bytes_rem;
    if (chunk > host_size) chunk = host_size;

    // Submit copy operation (explicit copy from device to host)
    auto d2h_copy_e = q.submit([&](handler &h) {
      // Range of buffer that needs to accessed
      auto buf_range = chunk / sizeof(unsigned long);
      // offset starts at 0 - incremented by chunk size each iteration
      auto buf_offset = offset / sizeof(unsigned long);
      // Access host_size range of buffer starting at buf_offset
      accessor mem(dev_buf, h, buf_range, buf_offset);
      // Reading from device buffer into host memory
      h.copy(mem, hostbuf);
    });

    // Wait for explicit copy from host memory to device buffer to complete
    d2h_copy_e.wait();

    // Get time for copy operation using Sycl event information (return in
    // nanoseconds)
    sum_time_ns += SyclGetQSubExecTimeNs(d2h_copy_e);  // Nanoseconds

    // **** Verification **** //

    // Verify data read back matches data that was written, if not increment
    // error count
    for (unsigned long i = 0; i < chunk / sizeof(unsigned long); ++i) {
      compare_count++;
      if (hostbuf[i] != (i + offset)) {
        ++errors;
        if (errors <= 32) {  // only print 32 errors
          std::cout << "Verification failure at element " << i << ", expected "
                    << i << " but read back " << hostbuf[i] << "\n";
        }
        chunk_errors++;
        if (chunk_errors <= 32) {  // only print 32 errors
          std::cout << "Verification failure at element " << i << "; chunk_cnt "
                    << chunk_cnt_rd << ";, expected 0x" << std::hex << i
                    << " \\ " << std::dec << i << " but read back 0x"
                    << std::hex << hostbuf[i] << " \\ " << std::dec
                    << hostbuf[i] << "\n";
        }
      }
    }  // End of for loop

    if (chunk_errors > 0) {
      std::cout << "chunk_errors for chunk " << chunk_cnt_rd << " was "
                << chunk_errors << " \\ 0x" << std::hex << chunk_errors
                << std::dec
                << "\n";  // Restoring manipulator to decimal in the end of cout
      chunk_errors = 0;   // reset for next chunk
    }

    // Increment offset and decrement remaining bytes by size of transfer
    offset += chunk;
    bytes_rem -= chunk;
    chunk_cnt_rd++;

  }  // End of read-from-device while loop

  // Report read bandwidth
  std::cout << (((float)max_alloc_size / kMB) /
                ((float)sum_time_ns / 1000000000))
            << " MB/s\n";

  // **** Report results from global memory test **** //

  std::cout << "Verifying data ...\n";

  if (errors == 0) {
    std::cout << "Successfully wrote and readback " << (max_alloc_size / kMB)
              << " MB buffer\n\n";
  } else {
    std::cout << "Wrote and readback " << (max_alloc_size / kMB)
              << " MB buffer\n";
    std::cout
        << "Failed write/readback test with " << errors << " errors out of "
        << compare_count << " \\ 0x" << std::hex << compare_count
        << std::dec  // Restoring manipulator to decimal at the end of cout
        << " comparisons\n\n";
  }

  // Free allocated host memory
  free(hostbuf);
  return errors;
}

//////////////////////////////////
// **** HostSpeed function **** //
//////////////////////////////////

// Inputs:
// queue &q - queue to submit operation
// Returns:
// 0 is test passes, 1 if test fails

// The function does the following tasks:
// 1. Test entire device global memory by writing and reading to it
// 2. Write to device global memory in smaller chunks and measure transfer time
// for each chunk
// 3. Read back data from device global memory and measure transfer time for
// each chunk
// 4. Verify data read back from device matches data written to device
// 5. Calculate write bandwidth, read bandwidth and read-write bandwidth from
// results of above transfers

// Following additional functions are used for the above tasks:
// More details about these funtions can be found with the corresponsing
// definitions
// 1. size_t TestGlobalMem(queue &q)
// 2. struct Speed WriteSpeed(queue &q, buffer<char,1> &device_buffer, char
// *hostbuf_wr, size_t block_bytes, size_t total_bytes)
// 3. struct Speed ReadSpeed(queue &q, buffer<char,1> &device_buffer, char
// *hostbuf_rd, size_t block_bytes, size_t total_bytes)
// 4. bool CheckResults
// 4. unsigned long SyclGetQSubExecTimeNs(event e)
// 5. unsigned long SyclGetTotalTimeNs(event first_evt, event last_evt)

int ShimMetrics::HostSpeed(queue &q) {
  // Total bytes to transfer
  constexpr size_t kMaxBytes = 8 * kMB;  // 8 MB;
  constexpr size_t kMaxChars = kMaxBytes / sizeof(char);

  // Block size of each transfer in bytes
  constexpr size_t kMinBytes = 32 * kKB;  // 32 KB
  size_t block_bytes = kMinBytes;

  // Call function to verify write to and read from the entire device global
  // memory
  if (ShimMetrics::TestGlobalMem(q) != 0) {
    std::cout << "Error: Global memory test failed\n";
    return 1;
  }

  // **** Device buffer **** //

  // Creating device buffer to span kMaxBytes size
  // Buffer that WriteSpeed and ReadSpeed functions write to & read from
  buffer<char, 1> device_buffer{range<1>{kMaxChars}};

  // **** Host memory allocation and initialization **** //

  // hostbuf_wr is used by WriteSpeed function to get input data to device
  // buffer hostbuf_rd is used by ReadSpeed function to store data read from
  // device buffer
  char *hostbuf_rd = (char *)malloc(kMaxBytes);
  char *hostbuf_wr = (char *)malloc(kMaxBytes);

  // Initializing input on host to be written to device buffer
  srand(time(NULL));
  // Create sequence: 0 rand1 ~2 rand2 4 ...
  for (size_t j = 0; j < kMaxChars; j++) {
    if (j % 2 == 0)
      hostbuf_wr[j] = (j & 2) ? ~j : j;
    else
      hostbuf_wr[j] = rand() * rand();
  }

  // *** Warm-up link before measuring bandwidth *** //

  // Writing to device buffer from initialized host memory
  WriteSpeed(q, device_buffer, hostbuf_wr, block_bytes, kMaxBytes);
  // Reading from device buffer into allocated host memory
  ReadSpeed(q, device_buffer, hostbuf_rd, block_bytes, kMaxBytes);

  // **** Block transfers to measure bandwidth **** //

  // Total number of iterations to write total bytes (i.e. kMaxBytes) in blocks
  // of block_bytes size
  size_t iterations = 1;
  for (size_t i = kMaxBytes / block_bytes; i >> 1; i = i >> 1) iterations++;

  // struct Speed in defined in hostspeed.hpp and is used to store transfer
  // times Creating array of struct to store output from each iteration The
  // values from each iteration are analyzed to report bandwidth
  struct Speed rd_bw[iterations];
  struct Speed wr_bw[iterations];

  // std::cout is manipulated to format output
  // Storing old state of std::cout to restore
  std::ios old_state(nullptr);
  old_state.copyfmt(std::cout);

  // Variable accumulate store result of each iteration
  bool result = true;

  // Iterate till total bytes (i.e. kMaxBytes) have been transferred
  // and accumulate results in rd_bw and wr_bw structs
  for (size_t i = 0; i < iterations; i++, block_bytes *= 2) {
    std::cout << "Transferring " << (kMaxBytes / kKB) << " KBs in "
              << (kMaxBytes / block_bytes) << " " << (block_bytes / kKB)
              << " KB blocks ...\n";
    wr_bw[i] = WriteSpeed(q, device_buffer, hostbuf_wr, block_bytes, kMaxBytes);
    rd_bw[i] = ReadSpeed(q, device_buffer, hostbuf_rd, block_bytes, kMaxBytes);
    // Verify value read back matches value that was written to device
    result &= CheckResults(hostbuf_wr, hostbuf_rd, kMaxChars);
    // Restoring old format after each CheckResults function call to print
    // correct format in current loop
    std::cout.copyfmt(old_state);
  }

  // **** Report results **** //
  // The write and read have already completed in the for loop with calls to
  // ReadSpeed, WriteSpeed functions The two for loops below simply format and
  // print the output for these tests

  // **** Report results from writes to device **** //

  // Restoring value of block_bytes as it changed in for loop above
  block_bytes = kMinBytes;

  // Fastest transfer value used in read-write bandwidth calculation
  float write_topspeed = 0;

  std::cout << "\nWriting " << (kMaxBytes / kKB)
            << " KBs with block size (in bytes) below:\n";
  std::cout << "\nBlock_Size Avg Max Min End-End (MB/s)\n";

  for (size_t i = 0; i < iterations; i++, block_bytes *= 2) {
    std::cout << std::setw(8) << block_bytes << " " << std::setprecision(2)
              << std::fixed << wr_bw[i].average << " " << wr_bw[i].fastest
              << " " << wr_bw[i].slowest << " " << wr_bw[i].total << " \n";
    if (wr_bw[i].fastest > write_topspeed) write_topspeed = wr_bw[i].fastest;
    if (wr_bw[i].total > write_topspeed) write_topspeed = wr_bw[i].total;
    // Restoring old format after each CheckResults function call to print
    // correct format in current loop
    std::cout.copyfmt(old_state);
  }

  // **** Report results from read from device **** //

  // Restoring value of block_bytes as it changed in for loop above
  block_bytes = kMinBytes;

  // Fastest transfer value used in read-write bandwidth calculation
  float read_topspeed = 0;

  std::cout << "\nReading " << (kMaxBytes / kKB)
            << " KBs with block size (in bytes) below:\n";
  std::cout << "\nBlock_Size Avg Max Min End-End (MB/s)\n";

  for (size_t i = 0; i < iterations; i++, block_bytes *= 2) {
    std::cout << std::setw(8) << block_bytes << " " << std::setprecision(2)
              << std::fixed << rd_bw[i].average << " " << rd_bw[i].fastest
              << " " << rd_bw[i].slowest << " " << rd_bw[i].total << " \n";
    if (rd_bw[i].fastest > read_topspeed) read_topspeed = rd_bw[i].fastest;
    if (rd_bw[i].total > read_topspeed) read_topspeed = rd_bw[i].total;
    // Restoring old format after each CheckResults function call to print
    // correct format in current loop
    std::cout.copyfmt(old_state);
  }

  h2d_rd_bw = read_topspeed;
  h2d_wr_bw = write_topspeed;
  h2d_rd_wr_bw = ((read_topspeed + write_topspeed) / 2);

  std::cout << "\nHost write top speed = " << std::setprecision(2) << std::fixed
            << write_topspeed << " MB/s\n";
  std::cout << "Host read top speed = " << read_topspeed << " MB/s\n\n";
  std::cout << "\nHOST-TO-MEMORY BANDWIDTH = " << std::setprecision(0)
            << ((read_topspeed + write_topspeed) / 2) << " MB/s\n\n";

  // Restoring old format after each CheckResults function call to print correct
  // format in current loop
  std::cout.copyfmt(old_state);

  // Fail if any iteration of the transfer failed
  if (!result) std::cout << "\nFAILURE!\n";

  // Free allocated host memory
  free(hostbuf_rd);
  free(hostbuf_wr);

  return (result) ? 0 : 1;
}

///////////////////////////////////
// **** HostRWTest function **** //
///////////////////////////////////

// Inputs:
// 1. queue &q - queue to submit operation
// 2. size_t dev_offset - device buffer offset (for transfer from aligned memory
// in device, the default value of this is 0) Returns: 0 is test passes,
// terminates program if test fails

// The function does the following tasks:
// 1. Creates a device buffer sized an odd number of bytes
// 2. Host memory allocation for both read and write with padding to prenent
// overflow during unaligned transfer
// 3. Increments pointer to host memory to make it an unaligned address
// 4. Writes data from this unaligned address to device global memory
// 5. Reads back data from device global memory to aligned host memory pointer
// 6. Verifies data read back matches data written
// Program terminates if verification fails

int ShimMetrics::HostRWTest(queue &q, size_t dev_offset) {
  // Bytes to transfer (1KB)
  constexpr size_t kMaxBytes_rw = kKB;

  // **** Device and host memory offsets to make them unaligned **** //

  // Device offset is passed from calling function

  // Selected host memory (for input to device) offset for unaligned transfer =
  // 5
  constexpr signed int kHostInOffset = 5;
  // Selected host memory (to read back from device) offset for read back = 8
  constexpr signed int kHostRdOffset = 8;

  std::cout << "--- Running host read write test with device offset "
            << dev_offset << "\n";

  // **** Device buffer creation **** //

  // Expanding device buffer size; odd number of bytes (i.e. +3) makes sure that
  // DMA is not aligned
  constexpr size_t kOddMaxBytes = kMaxBytes_rw + 3;
  // Device buffer of kOddMaxbytes
  buffer<char, 1> dev_buf(range<1>{kOddMaxBytes});

  // **** Host memory allocation and initialization **** //

  // Padding both host memory blocks with extra space to avoid overflow during
  // unaligned transfers

  // Allocating host memory for input to device buffer
  char *host_in_buf = (char *)malloc(
      kMaxBytes_rw +
      (2 * kHostInOffset));  // Padding on both sides of memory, hence increment
                             // size by 2 * host_in_offset
  // Save original memory address before offset
  char *host_in_buf_ptr = host_in_buf;

  // Initialize host memory with some invalid data
  char invalid_host_input = -6;
  for (size_t j = 0; j < ((kMaxBytes_rw + (2 * kHostInOffset)) / sizeof(char));
       j++) {
    host_in_buf[j] = invalid_host_input;
  }
  // Increment pointer to make input host memory pointer non-aligned to test
  // un-aligned host ptr to un-aligned device ptr transfer
  host_in_buf += kHostInOffset;

  // Allocating host memory to read back into from device buffer
  char *host_rd_buf = (char *)malloc(
      kMaxBytes_rw +
      (2 * kHostRdOffset));  // Padding on both sides of memory, hence increment
                             // size by 2 * host_rd_offset
  // Save original memory address before offset
  char *host_rd_buf_ptr = host_rd_buf;

  // Initialize host memory with some invalid data
  char invalid_read_back = -3;
  for (size_t j = 0; j < ((kMaxBytes_rw + (2 * kHostRdOffset)) / sizeof(char));
       j++) {
    host_rd_buf[j] = invalid_read_back;
  }
  // Increment pointer to make read back memory pointer aligned to test reading
  // from un-aligned device ptr to aligned host ptr
  host_rd_buf += kHostRdOffset;

#ifdef DEBUG
  std::cout << "host input buf = " << host_in_buf << " , "
            << "host read back buf = " << host_rd_buf << "\n";
#endif

  srand(0);

  // **** Write to and read from device global memory **** //

  // Non-DMA read/writes of all sizes upto 1024
  for (size_t i = 1; i <= kMaxBytes_rw; i++) {
#ifdef DEBUG
    std::cout << "Read/write of " << i << " bytes\n";
#endif

    // host will have unique values for every write
    for (size_t j = 0; j < i; j++) {
      host_in_buf[j] = (char)rand();
    }

    // **** Write to device global memory **** //

    // Submit copy operation (explicit copy from host to device)
    auto h2d_copy_e = q.submit([&](handler &h) {
      // Range of buffer that needs to accessed = i bytes (chars)
      // Device buffer is accessed at dev_offset
      // Using +3 offset on aligned device pointer ensures that DMA is never
      // used (because the host ptr is not aligned)
      accessor mem(dev_buf, h, i, dev_offset);
      // Writing from host memory to device buffer
      h.copy(host_in_buf, mem);
    });
    // Wait for copy to complete
    h2d_copy_e.wait();

    // **** Read from device global memory **** //

    // Submit copy operation (explicit copy from device to host)
    auto d2h_copy_e = q.submit([&](handler &h) {
      // Range of buffer that needs to accessed = i bytes (chars)
      // Device buffer is accessed at dev_offset
      accessor mem(dev_buf, h, i, dev_offset);
      // Reading from device buffer into host memory
      h.copy(mem, host_rd_buf);
    });
    // Wait for copy to complete
    d2h_copy_e.wait();

// Verify values read back match the input
#ifdef DEBUG
    std::cout << host_rd_buf[0] << " , " << host_in_buf[0] << "\n";
#endif

    if (memcmp(host_in_buf, host_rd_buf, i) != 0) {
      std::cout << i << " bytes read/write FAILED!\n";
      for (size_t m = 0; m < i; m++) {
        if (host_in_buf[m] != host_rd_buf[m]) {
          std::cout << "char #" << m
                    << " , host input buffer = " << host_in_buf[m]
                    << " , host read back buffer = " << host_rd_buf[m] << "\n";
        }
      }
      assert(0);
    }

    // make sure bounds on both ends of buffer are ok
    for (signed int k = (-1 * kHostRdOffset); k < kHostRdOffset; k++) {
      if (k < 0)
        assert(host_rd_buf[k] == invalid_read_back);
      else
        assert(host_rd_buf[i + k] == invalid_read_back);
    }

  }  // End of for loop for writing/reading all bytes

  // Free allocated host memory
  free(host_in_buf_ptr);
  free(host_rd_buf_ptr);

  // Unaligned transfers completed successfully, test passed
  h2d_rw_test = true;
  return 0;
}

//////////////////////////////////////
// **** KernelClkFreq function **** //
//////////////////////////////////////

// Inputs:
// queue &q - queue to submit operation
// bool report_chk - control if Quartus compiled frequency should be read
// Returns:
// 0 is test passes, 1 if test fails

// The function does the following tasks:
// 1. Launches an no-operation NDRange kernel with global size 128 MB
// 2. Measures time taken for the above NDRange kernel using Sycl event
// profiling information
// 3. Obtain kernel clock frequency based on time take for 128 Mglobal
// operations (NDRange)
// 4. If the <report_chk> is true, compare the above hardware frequency with
// Quartus compiled frequency NOTE: This needs the reports or
// board_test.prj/reports directory from compilation output to be in the same
// directory as board_test.fpga binary
// 5. Return 0 (test pass) if measured frequency is within 2 of Quartus compiled
// frequency, else report error and terminate test NOTE: If <report_chk> is set
// to false, comparison with Quartus compiled frequency is not done and
// remaining tests in board_test continue without this frequency check of 2%
// error tolerance

int ShimMetrics::KernelClkFreq(queue &q, bool report_chk) {
  std::cout << "*** NOTE ***: This test expects the reports folder generated " 
            << "during compile to be in one of the following locations : \n"
            << "    1. Same directory as board_test.fpga binary\n"
            << "    2. Inside board_test.prj folder in the same directory as "
            << "board_test.fpga binary\n"
            << "    If the reports folder is not found, the test will return "
            << "only the measured kernel clock frequency and fail, none of the "
            << "other tests will run as hardware frequency may not be the expected "
            << "value and may lead to functional errors.\n"
            << "    If you wish to override this failure, please set "
            << "\"report_chk\" variable to \"false\" in <board_test.cpp> and "
            << "recompile host code only using \"-reuse-exe=board_test.fpga\" "
            << "option in compile command.\n"
            << "    Please run complete board_test at least once and ensure "
            << "the hardware frequency matches expected frequency, mismatch "
            << "may lead to functional error.\n\n";

  // **** Launching an empty kernel (no op) **** //

  // ND Range of kernel to launch
  constexpr size_t kTotalBytes =
      128 * kMB;  // 128 MB - this is the min amount known to be available on
                  // device memory (minimum on device - e.g. Cyclone V)
  constexpr size_t kGlobalSize = kTotalBytes / (sizeof(unsigned));

  auto e = q.submit([&](handler &h) {
    // Global range (1 dimension)
    constexpr size_t kN = kGlobalSize;
    // Work group Size (1 dimension)
    constexpr size_t kReqdWgSize = 32 * kKB;  // 32 KB
    h.parallel_for<NopNDRange>(
        nd_range<1>(range<1>(kN), range<1>(kReqdWgSize)), [=](auto id) 
		[[cl::reqd_work_group_size(1, 1, kReqdWgSize)]]{});
  });
  // Wait for operation to complete
  e.wait();

  // **** Get time for kernel event **** //

  float time = SyclGetQSubExecTimeNs(e);
  kernel_freq = ((float)kGlobalSize) /
                (time / 1000.0f);  // Time is returned in nanoseconds,
                                   // converting global size to Mega and ns to s
  // This bandwidth is used in kernel_mem_bw.cpp
  // bandwidth_nop = kernel_freq * sizeof(unsigned);

  // **** Compare measured clock frequency with Quartus Prime compiled fmax **** //

  // Check Quartus reports if user has selected this (true by default)
  if (report_chk) {
    // Pointer to reports directory
    DIR *reports_dir;

    // Look for reports folder in the same directory as binary first,
    // if not found look inside board_test.prj
    const char *paths[] = {"reports/lib/json/",
                           "board_test.prj/reports/lib/json/"};
    // Variable to track if reports directory is found
    bool dir_found = false;
    std::string dir_name;

    for (size_t i = 0; i < 2; i++) {
      if ((reports_dir = opendir(paths[i]))) {
        dir_name = paths[i];
        dir_found = true;
        // Do not run for loop further once valid path found as this will change
        // reports_dir pointer to next value in paths array
        break;
      }
    }

    if (!dir_found) {
      // Directory not found, report and terminate test
      std::cout
          << "Failed to find reports directory, cannot locate Quartus compiled "
          << "frequency. Please see Note at the beginning of this kernel clock "
          << "frequency test.\n"
          << "Reporting the measured frequency and terminating this test.\n\n"
          << "Measured Frequency = " << kernel_freq << " MHz.\n";
      return 1;
    } else {
      // reports directory found, look for quartus.json file
      // Pointer to directory entries
      struct dirent *dir_entry;

      // Look for quartus.json inside reports/lib/json
      std::string file_name = "quartus.json";

      // The following lookup_tag is specific to the quartus.json generated by
      // oneAPI compile
      std::string lookup_tag = "\"kernel clock fmax\" : ";
      // Variable to track if quartus.json is found
      bool file_found = false;
      // Variable to track if kernel clock frequency tag is found
      bool tag_found = false;

      // Iterate through directory contents till quartus.json is found
      while ((dir_entry = readdir(reports_dir)) != NULL) {
        if (dir_entry->d_name == file_name) {
          // Open file for reading
          std::ifstream json_fs{(dir_name + dir_entry->d_name)};
          std::string rd_line;

          // Check if file opened
          if (json_fs) {
            float quartus_fmax = 0.0;
            file_found = true;

            // Read each line in file till lookup_tag is found
            while (std::getline(json_fs, rd_line)) {
              // Look for the fmax tag
              if (rd_line.find(lookup_tag) != std::string::npos) {
                tag_found = true;
                // The look-up tag format is "kernel clock fmax" :
                // "freq-val-in-float"
                size_t st_pos =
                    rd_line.find(lookup_tag) + lookup_tag.size() + 1;
                // Extract freq-val-in-float starting at st_pos
                std::string fmax_tmp = rd_line.substr(st_pos);
                size_t end_pos = fmax_tmp.find("\"");
                // Remove the quotes (") after the freq-val-in-float at end_pos
                quartus_fmax = std::stof(fmax_tmp.substr(0, end_pos));

                // No need to iterate through rest of the file if fmax tag is
                // found
                break;

              }  // End of if extracting fmax from rd_line

            }  // End of while reading each line from the json file

            // Error and exit if lookup_tag not found
            if (!tag_found) {
              std::cout
                  << "Cannot find Quartus compiled fmax in located "
                  << file_name
                  << " , please ensure full kernel compile has run and "
                  << "hardware generation completed successfully.\n"
                  << "Failed to report Quartus compiled frequency, please see "
                  << "NOTE at the beginning of this kernel clock frequency "
                  << "test.\n"
                  << "Reporting measured frequency and terminating test.\n\n"
                  << "Measured Frequency = " << kernel_freq << "\n";
              closedir(reports_dir);
              // No need to iterate in directory once report is found and no
              // fmax found
              return 1;
            } else {
              // Quartus compiled frequency found, report it
              std::cout << "Measured Frequency    =   " << kernel_freq
                        << " MHz \n";
              std::cout << "Quartus Compiled Frequency  =   " << quartus_fmax
                        << " MHz \n\n";

              // Check that hardware frequency is within 2% of Quartus compiled
              // frequency, terminate test if its not
              float PercentError =
                  (fabs(quartus_fmax - kernel_freq) / (quartus_fmax)) * 100;
              if (PercentError < 2)
                std::cout << "Measured Clock frequency is within 2 percent of "
                          << "quartus compiled frequency. \n";
              else {
                std::cout << "\nError: measured clock frequency not within 2 "
                          << "percent of quartus compiled frequency. \n"
                          << "Terminating test.\n";
                closedir(reports_dir);
                return 1;
              }
            }  // End of if - else based on tag_found value
          } else {
            // quartus.json could not be opened, report error and terminate test
            std::cout
                << "Failed to open " << file_name << "\n"
                << "Failed to report Quartus compiled frequency, please see "
                << "NOTE at the beginning of this kernel clock frequency "
                << "test.\n"
                << "Reporting measured frequency and terminating test.\n\n"
                << "Measured Frequency = " << kernel_freq << "\n";
            closedir(reports_dir);
            return 1;
          }  // End of if - else reading the json file

          // No need to iterate through rest of directory once required file is
          // found
          break;

        }  // End of if looking for quartus.json

      }  // End of while reading file names inside reports/lib/json directory

      if (!file_found) {
        // Could not find quartus.json, report error and terminate test
        std::cout
            << "Cannot find " << file_name << " JSON file in the " << paths[0]
            << " or " << paths[1] << "directory\n"
            << "Failed to report Quartus compiled frequency, please see NOTE "
            << "at the beginning of this kernel clock frequency test.\n"
            << "Reporting measured frequency and terminating test.\n\n"
            << "Measured Frequency = " << kernel_freq << "\n";
        closedir(reports_dir);
        return 1;
      }  // End of if reporting file not found

      // Successfully read Quartus compiled frequency, close reports directory
      closedir(reports_dir);

    }  // End of if - else to read data from reports directory
  } else {
    // USer has selected to not to compare measured frequency with Quartus
    // compiled frequency, report and continue other tests
    std::cout
        << "*** NOTE ***: User has selected to turn off the comparison of "
        << "measured frequency with Quartus compiled frequency by setting the "
        << "\"report_chk\" variable to \"false\" in <board_test.cpp>\n"
        << "The Quartus compiled frequency will not be reported and the "
        << "remaining tests will run without this check\n"
        << "Please see Note at the beginning of this kernel clock frequency "
        << "test for more information about this setting.\n"
        << "Reporting measured frequency and continuing remaining tests.\n"
        << "Measured Frequency = " << kernel_freq << "\n";
  }  // End of if - else to check for reports

  return 0;
}

//////////////////////////////////////
// **** KernelLatency function **** //
//////////////////////////////////////

// Inputs:
// queue &q - queue to submit operation
// Returns:
// 0 indicating successful completion (no error checks)

// The function does the following tasks:
// 1. Launch large number of no operation kernels
// 2. Measure total time for the above kernels to launch and finish
// 3. Calculate kernel latency and kernel throughput based on total time and
// number of kernels
// 4. Report the latency and throughput and return

int ShimMetrics::KernelLatency(queue &q) {
  // **** Launch no-op kernel multiple times **** //
  auto start = std::chrono::system_clock::now();
  constexpr size_t kNumKernels = 10000;
  for (size_t l = 0; l < kNumKernels; l++) {
    auto e = q.submit([&](handler &h) {
      h.single_task<NopSingleTask>([=]() {});  // no operation kernel
    });
  }
  // Wait for all queued tasks to finish
  q.wait();
  auto stop = std::chrono::system_clock::now();

  // Time taken for kNumKernels to launch and finish
  std::chrono::duration<float> time = (stop - start);  // in seconds

  // **** Report kernel latency **** //

  // Storing old state of std::cout to restore after output printed
  std::ios old_state(nullptr);
  old_state.copyfmt(std::cout);

  // Calculating throughput in kernels/ms, averaged over kNumKernels launches
  kernel_thruput = kNumKernels * 1 / (time.count() * 1000.0f);
  kernel_latency = (time.count() * 1000000.0f / kNumKernels);
  std::cout << "Processed " << kNumKernels << " kernels in "
            << std::setprecision(4) << std::fixed << (time.count() * 1000.0f)
            << " ms\n";
  std::cout << "Single kernel round trip time = " << kernel_latency << " us\n";
  std::cout << "Throughput = " << kernel_thruput << " kernels/ms\n";

  // Restoring old format after each check_results function call to print
  // correct format in current loop
  std::cout.copyfmt(old_state);
  std::cout << "Kernel execution is complete\n";

  // Test complete
  return 0;
}

/////////////////////////////////////////
// **** KernelLaunchTest function **** //
/////////////////////////////////////////

// Inputs:
// queue &q - queue to submit operation
// Returns:
// 0 is test passes, 1 if test fails

// The function does the following tasks:
// 1. Create a pipe between 2 kernels (sender kernel and receiver kernel)
// 2. Launch sender kernel
// 3. Sender kernel writes a known value (kTestValue) to the pipe
// 4. Launch receiver kernel
// 5. Receiver kernel reads the value from pipe and writes to memory
// 6. Host reads data back from memory and checks if the value read is equal to
// the known value (kTestVakue) Test fails if there is a data mismatch

int ShimMetrics::KernelLaunchTest(queue &q) {
  // Value to be written to pipe from KernelSender
  constexpr unsigned int kTestValue = 0xdead1234;

  // Create device buffer to read back data
  std::array<unsigned int, 1> init_val = {0};
  buffer dev_buf(init_val);

  // **** Launch sender kernel (writes to pipe) **** //

  std::cout << "Launching kernel KernelSender ...\n";
  auto e_send = q.submit([&](handler &h) {
    // Global range (1 dimension)
    constexpr size_t kN = 1;
    // Work group size (1 dimension)
    constexpr size_t kReqdWgSize = 1;
    h.parallel_for<KernelSender>(
        nd_range<1>(range<1>(kN), range<1>(kReqdWgSize)), [=](auto id) {
          SendertoReceiverPipe::write(kTestValue);  // Blocking write
        });
  });

  // **** Launch receiver kernel (reads from pipe) **** //

  std::cout << "Launching kernel KernelReceiver ...\n";
  auto e_receive = q.submit([&](handler &h) {
    // Global range (1 dimension)
    constexpr size_t kN = 1;
    // Work group size (1 dimension)
    constexpr size_t kReqdWgSize = 1;
    accessor mem(dev_buf, h);
    h.parallel_for<KernelReceiver>(
        nd_range<1>(range<1>(kN), range<1>(kReqdWgSize)), [=](nd_item<1> it) {
          // Initialize to 0
          unsigned int pipe_value = 0;
          // Blocking read from pipe
          pipe_value = SendertoReceiverPipe::read();
          auto gid = it.get_global_id(0);
          mem[gid] = pipe_value;
        });
  });

  // **** Wait for sender and receiver kernels to finish **** //

  std::cout << "  ... Waiting for sender\n";
  e_send.wait();
  std::cout << "Sender sent the token to receiver\n";
  std::cout << "  ... Waiting for receiver\n";
  e_receive.wait();

  // Read back data written by pipe to device memory
  host_accessor h_buf_access{dev_buf};
  if (h_buf_access[0] != kTestValue) {
    std::cout << "Kernel Launch Test failed, incorrrect value read back from "
              << "pipe between sender and receiver kernel:\n"
              << "Value written to pipe : " << kTestValue
              << " Value read back : " << h_buf_access[0] << "\n";
    return 1;
  }

  return 0;
}

////////////////////////////////////
// **** KernelMemRW function **** //
////////////////////////////////////

// Inputs:
// queue &q - queue to submit operation
// Returns:
// 0 is test passes, 1 if device memory allocation is 0 or if test fails

// The function does the following tasks:
// 1. Gets maximum allocation size for device global memory
// 2. Calculates number of unsigned type elements that can be written to device
// and allocates device buffer to span this size
// 3. Allocates host memory for input & output to & from device global memory
// 4. Write initial data to entire device global memory
// 5. Launches kernel to modify the values written to device global memory
// 6. Reads the modified data from device global memory into host memory
// 7. Verifies data matches expected value

// Following additional function is used in this test:
// This is defined in this file and declared in corresponding header
// (kernel_mem_rw.hpp) More details can be found with the corresponsing
// definition
// 1. void InitializeVector(unsigned *vector, size_t size, size_t offset)

int ShimMetrics::KernelMemRW(queue &q) {
  // Test fails if max alloc size is 0
  if (max_alloc_size == 0) {
    std::cout << "Maximum global memory allocation supported by Sycl device is "
              << "0! Cannot run kernel-to-memory read wite test\n\n";
    return 1;
  }

  std::cout << "Maximum device global memory allocation size is "
            << max_alloc_size << " bytes \n";

  // Number of integer type vectors supported on the device
  size_t max_dev_vectors = max_alloc_size / sizeof(unsigned);

  // **** Host memory Allocation **** //

  // Allocate host vectors
  // ND Range kernel uses max_dev_vectors to calculate the global range...
  // ...SYCL ID queries are expected to fit within MAX_INT,...
  // ...limit the range to prevent trunating data and errors
  size_t num_host_vectors = (max_dev_vectors > INT_MAX) ? kGB : max_dev_vectors;
  size_t host_vector_size_bytes = num_host_vectors * sizeof(unsigned);

  // Host memory used for storing input data to device
  unsigned *host_data_in = (unsigned *)malloc(host_vector_size_bytes);
  // Host memory used to store data read back from device
  unsigned *host_data_out = (unsigned *)malloc(host_vector_size_bytes);

  // The below while loop checks if host memory allocation failed and tries to
  // allocate a smaller chunk if above allocation fails
  while (host_data_in == NULL || host_data_out == NULL) {
    num_host_vectors = num_host_vectors / 2;
    host_vector_size_bytes = num_host_vectors * sizeof(unsigned);
    host_data_in = (unsigned *)malloc(host_vector_size_bytes);
    host_data_out = (unsigned *)malloc(host_vector_size_bytes);
  }

  std::cout << "Finished host memory allocation for input and output data\n";

  // **** Device buffer creation **** //

  std::cout << "Creating device buffer\n";

  buffer<unsigned, 1> dev_buf(range<1>{max_dev_vectors});

  // **** Writing to device global memory **** //

  // If max_dev_vectors (i.e. device global memory allocation size) >
  // MAX_INT,...
  // ... multiple iterations/writes to device memory are needed to fill entire
  // global memory with preset data ...
  // ... To ensure full device global memory is written to (even if it is evenly
  // distributable by kGB) - ...
  // ... Pad with (kGB - 1) before dividing by kGB.
  size_t num_dev_writes =
      (max_dev_vectors + kGB - 1) /
      kGB;  // Calculating number of writes needed (adding kGB - 1 to prevent
            // missing out few bytes of global address space due to rounding

  // Access device memory in chunks and initialize it
  for (size_t vecID = 0; vecID < num_dev_writes; vecID++) {
    // Each iteration writes kGB size chunks, so offset increments by lGB
    size_t global_offset = vecID * kGB;
    size_t current_write_size = kGB;

    // Remaining vectors for last set (calculated this way as the padding may
    // make the last write bigger than actual global memory size on buffer)
    if (vecID == (num_dev_writes - 1)) {
      current_write_size = max_dev_vectors - global_offset;
    }

    // If host buffer - host_data_in is smaller than max_dev_vectors, transfer
    // data in multiple writes
    size_t offset_bytes = 0;
    size_t bytes_rem = current_write_size * sizeof(unsigned);

    while (bytes_rem > 0) {
      size_t chunk = bytes_rem;
      // chunk is greater than host buffer size, break into smaller chunks
      if (chunk > host_vector_size_bytes) chunk = host_vector_size_bytes;
      // Number of elements written in 1 transfer (copy operation)
      num_host_vectors = chunk / sizeof(unsigned);
      // Offset if max_dev_vectors chunk is broken into smaller portions
      size_t offset = offset_bytes / sizeof(unsigned);
      // Initialize input data on host with the total offset value
      InitializeVector(host_data_in, num_host_vectors,
                       (global_offset + offset));
      // Submit copy operation (explicit copy from host to device)
      auto h2d_copy_e = q.submit([&](handler &h) {
        // Range of buffer that needs to accessed is num_host_vectors
        // offset starts at 0 - incremented by chunk size each iteration
        auto buf_offset = global_offset + offset;
        accessor mem(dev_buf, h, num_host_vectors, buf_offset);
        // Writing from host memory to device buffer
        h.copy(host_data_in, mem);
      });
      // Wait for copy operation to complete
      h2d_copy_e.wait();
      // Increment offset and decrement remaining bytes by chunk size each
      // interation
      offset_bytes += chunk;
      bytes_rem -= chunk;
    }  // End of write while loop
  }    // End of write for loop

  std::cout << "Finished writing to device buffers \n";

  // **** Submitting kernel operation **** //

  std::cout << "Launching kernel MemReadWriteStream ... \n";

  // Enqueue kernel to access all of global memory
  // Multiple enqueues are needed to access the whole ...
  // ... global memory address space from the kernel ...
  // ... if max_dev_vectors > kGB

  for (size_t vecID = 0; vecID < num_dev_writes; vecID++) {
    // Each iteration writes kGB size chunks, so offset increments by kGB
    size_t global_offset = vecID * kGB;
    size_t current_write_size = kGB;

    // Remaining vectors for last set (calculated this way as the padding may
    // make the last read bigger than actual global memory size on buffer)
    if (vecID == (num_dev_writes - 1)) {
      current_write_size = max_dev_vectors - global_offset;
    }

    std::cout << "Launching kernel with global offset : " << global_offset
              << "\n";

    // launch kernel
    auto e = q.submit([&](handler &h) {
      // Global range (1 dimension)
      size_t N = current_write_size;
      accessor mem(dev_buf, h, N, global_offset);
      h.parallel_for<MemReadWriteStream>(range<1>{N}, [=](item<1> it) {
        // Add 2 to all data read from global memory (meaning adding 2 to all
        // the offsets calculated in write loops above)
        auto gid = it.get_id(0);
        mem[gid] = mem[gid] + 2;
      });
    });
  }  // End of kernel launch for loop

  // Wait for operation to complete
  q.wait();

  std::cout << "... kernel finished execution. \n";

  // **** Read back data from device global memory & verify **** //

  for (size_t vecID = 0; vecID < num_dev_writes; vecID++) {
    // Each iteration writes kGB size chunks, so offset increments by kGB
    size_t global_offset = vecID * kGB;
    size_t current_read_size = kGB;

    // Remaining vectors for last set (calculated this way as the padding may
    // make the last read bigger than actual global memory size on buffer)
    if (vecID == (num_dev_writes - 1)) {
      current_read_size = max_dev_vectors - global_offset;
    }

    // If host buffer - host_data_out is smaller than max_dev_vectors, transfer
    // data in multiple reads
    size_t bytes_rem = current_read_size * sizeof(unsigned);
    size_t offset_bytes = 0;

    while (bytes_rem > 0) {
      size_t chunk = bytes_rem;
      // chunk is greater than host buffer size, break into smaller chunks
      if (chunk > host_vector_size_bytes) chunk = host_vector_size_bytes;
      // Number of elements read in 1 transfer (copy operation)
      num_host_vectors = chunk / sizeof(unsigned);
      // Offset if max_dev_vectors chunk is broken into smaller portions
      size_t offset = offset_bytes / sizeof(unsigned);
      // Submit copy operation (explicit copy from device to host)
      auto d2h_copy_e = q.submit([&](handler &h) {
        // Range of buffer that needs to accessed is num_host_vectors
        // offset starts at 0 - incremented by chunk size each iteration
        auto buf_offset = global_offset + offset;
        accessor mem(dev_buf, h, num_host_vectors, buf_offset);
        // Reading from device buffer into host memory
        h.copy(mem, host_data_out);
      });
      // Wait for copy operation to complete
      d2h_copy_e.wait();

      // **** Verify output **** //

      // Compare value read back is offset + 2
      // initial value written was offset, incremented by 2 in kernel
      for (size_t i = 0; i < num_host_vectors; i++) {
        if (host_data_out[i] != (unsigned)(global_offset + offset + i + 2)) {
          std::cout << "Verification failed " << i << " : " << host_data_out[i]
                    << " != " << ((unsigned)(global_offset + offset + i + 2))
                    << "\n";
          // Free host memory and return if verification fails
          if (host_data_in) free(host_data_in);
          if (host_data_out) free(host_data_out);
          return 1;
        }
      }
      // Increment offset and decrement remaining bytes by chunk size each
      // interation
      offset_bytes += chunk;
      bytes_rem -= chunk;
    }
  }  // End of read for loop

  // All values verified successfully - test passed
  std::cout << "Finished Verification\n";

  // Free allocated host memory
  if (host_data_in) free(host_data_in);
  if (host_data_out) free(host_data_out);

  std::cout << "KERNEL TO MEMORY READ WRITE TEST PASSED \n";
  return 0;
}

////////////////////////////////////
// **** KernelMemBW function **** //
////////////////////////////////////

// Inputs:
// queue &q - queue to submit operation
// Returns:
// 0 if test passes, 1 if device max allocation size is 0 or verification fails

// The function does the following tasks:
// 1. Get max allocation size for device global memory, limit device buffer size
// to 4 GB if the max alloc is greater
// 2. Read board_spec.xml (if OFS_OCL_SHIM_ROOT_HW is set) to get number of
// memory interfaces/banks
// 3. Allocate host memory and initialize with random values
// 4. Write the data from host memory to device global memory (initializing
// device global memory with random values)
// 4. Launch 3 kernels for each dimm/memory bank:
//      a. MemWriteStream - Write to device global memory
//      b. MemReadStream - Read from device global memory
//      c. MemReadWriteStream - Read, modify and write to device global memory
// Each of the kernel does this for each dimm (test assumes max of 8 dimms), if
// number of dimms is less, the kernel read/write defaults to lowest memory bank
// (1)
// 5. Calculate bandwidth for read, write and read-write based on time taken for
// each of the operation above
// 6. Read the theoretical bandwidth from board_spec.xml, calculate utilization
// and report results

int ShimMetrics::KernelMemBW(queue &q) {
  std::cout << "Note: This test assumes that design was compiled with "
            << "-Xsno-interleaving option\n\n";

  // Test fails if max alloc size is 0
  if (max_alloc_size == 0) {
    std::cout << "Maximum global memory allocation supported by Sycl device is "
              << "0! Cannot run kernel-to-memory bandwidth test\n\n";
    return 1;
  }

  // Default number of memory banks/DIMMs in the OpenCL/oneAPI shim (assumed to
  // prevent test from failing if board_spec.xml data cannot be read)
  constexpr size_t kDefaultNumBanks = 8;

  // If device global memory > 4 GB , limit the transfer size to 4 GB for this
  // test
  size_t total_bytes_used =
      (max_alloc_size > (4 * kGB)) ? (4 * kGB) : max_alloc_size;

  // Transfer size in number of unsigned elements
  size_t vector_size = total_bytes_used / sizeof(unsigned);

  // **** Read the number of memory banks from board_spec.xml **** //

  // Variable to track if board_spec.xml is found
  bool xml_found = false;
  // Variable to track if required environment variable set by user
  bool env_var = false;

  size_t num_banks = 0;

  // Get shim directory path from environment variable
  char *SHIM_PATH = (char *)malloc(10000 * sizeof(char));
  if ((std::getenv("OFS_OCL_SHIM_ROOT_HW")) != NULL) {
    strcpy(SHIM_PATH, (std::getenv("OFS_OCL_SHIM_ROOT_HW")));
    env_var = true;
  } else {
    std::cout << "\nFailed to get OFS_OCL_SHIM_ROOT_HW environment variable\n";
    // XML data could not be read, setting value to default of 8
    num_banks = kDefaultNumBanks;
  }

  if ((SHIM_PATH != NULL) && (env_var == true)) {
    // Directory path obtained successfully, parse board_spec.xml
    char app_path[] = "/board_spec.xml";
    char *file_path = (char *)malloc(
        10000 *
        sizeof(char));  
    strcpy(file_path, strcat(SHIM_PATH, app_path));

    // Try opening board_spec.xml
    FILE *board_sp = std::fopen(file_path, "r");
    if (!board_sp) {
      // Failed to open board_spec.xml, xml_found variable will stay false
      std::cout
          << "Error in opening board spec.xml. \n"
          << "Cannot read memory bank information from board_spec.xml file,"
          << "this test will run without reporting the global memory "
          << "information from board_spec.xml.\n";
      // XML data could not be read, setting value to default of 8
      num_banks = kDefaultNumBanks;
    } else {
      // board_spec.xml found, check the sizes of the memory banks in global
      // memory interface
      xml_found = true;
      char *size_of_bank = (char *)malloc(100 * sizeof(char));
      while (std::feof(board_sp) == 0) {
        char *temp_data = (char *)malloc(100 * sizeof(char));
        fscanf(board_sp, "%s", temp_data);
        // The board_spec.xml used <interface> tag inside <global_memory> tag
        // for memory banks
        if (strcmp(temp_data, "<interface") == 0) {
          while (!std::feof(board_sp)) {
            char *temp_rd_data = (char *)malloc(100 * sizeof(char));
            fscanf(board_sp, "%s", temp_rd_data);
            // If the end tag for <global_memory> is found, all memory interface
            // sizes read, break from loop
            if (strncmp(temp_rd_data, "</global_mem>", 13) == 0) {
              free(temp_rd_data);
              break;
            } else if (strncmp(temp_rd_data, "size", 4) == 0) {
              // report the size of this memory bank (size attribute in
              // interface tag)
              num_banks++;
              strcpy(size_of_bank, (strtok(temp_rd_data, "=\"")));
              strcpy(size_of_bank, (strtok(NULL, "=\"")));
              std::cout << "Size of memory bank " << num_banks << " = "
                        << size_of_bank << " bytes \n";
            }
            free(temp_rd_data);
          }
          free(temp_data);
          break;  // No need to iterate through rest of the file once interface
                  // is found
        }
        free(temp_data);  // If interface not found, reaches this point

      }  // End of while reading board_spec.xml

      // Close board_spec.xml
      if (std::fclose(board_sp) != 0)
        std::cout << "Error closing board_spec.xml\n\n";
      free(size_of_bank);

    }  // End of if - else checking board_spec.xml open

    free(file_path);
  }

  // **** Host memory allocation & initialization **** //

  // Host memory used to store input data to device buffer
  unsigned *host_data_in = (unsigned *)malloc(total_bytes_used);
  // Host memory used to store data read back from device
  unsigned *host_data_out = (unsigned *)malloc(total_bytes_used);

  // The below while loop checks if hostbuf allocation failed and tries to
  // allocate a smaller chunk if above allocation fails
  while (host_data_in == NULL || host_data_out == NULL) {
    vector_size = vector_size / 2;
    total_bytes_used = vector_size * sizeof(unsigned);
    host_data_in = (unsigned *)malloc(total_bytes_used);
    host_data_out = (unsigned *)malloc(total_bytes_used);
  }

  // Initialize host memory
  InitializeVector(host_data_in, vector_size);
  InitializeVector(host_data_out, vector_size);

  // **** Write data to device & launch kernels **** //

  std::cout << "\nPerforming kernel transfers of " << (total_bytes_used / kMB)
            << " MBs on the default global memory (address starting at 0)\n";

  // The loop launches 3 different kernels to measure:
  // kernel to memory write bandwidth using "MemWriteStream" kernel
  // kernel to memory read bandwidth using "MemReadStream" kernel
  // kernel to memory read-write bandwidth using "MemReadWriteStream" kernel
  // const char *kernel_name[] = {"MemWriteStream", "MemReadStream",
  // "MemReadWriteStream"};
  constexpr size_t kNumKernels = 3;
  std::string kernel_name[kNumKernels] = {"MemWriteStream", "MemReadStream",
                                          "MemReadWriteStream"};

  // Array used to store the bandwidth measurement for each kernel for each
  // memory bank
  std::vector<std::vector<float> > bw_kern;

  // k = 0 launches kernel_name[0] i.e. MemWriteStream kernel
  // k = 1 launches kernel_name[1] i.e. MemReadStream kernel
  // k = 2 launches kernel_name[2] i.e. MemReadWriteStream kernel

  for (unsigned k = 0; k < kNumKernels; k++) {
    std::cout << "Launching kernel " << kernel_name[k] << " ... \n";

    std::vector<float> bw_bank;

    // Launch each kernel once for each memory bank
    for (unsigned b = 0; b < num_banks; b++) {
      // Assign a memory channel for each transfer (needed for multi-bank
      // OpenCL/oneAPI shims) default memory channel is 1 (lowest)
      property_list buf_prop_list{property::buffer::mem_channel{1}};

      switch (b) {
        // if the board_spec.xml has fewer banks than the dimms, mem_channel
        // defaults to 1
        case 0:
          buf_prop_list = {property::buffer::mem_channel{1}};
          break;
        case 1:
          buf_prop_list = {property::buffer::mem_channel{2}};
          break;
        case 2:
          buf_prop_list = {property::buffer::mem_channel{3}};
          break;
        case 3:
          buf_prop_list = {property::buffer::mem_channel{4}};
          break;
        case 4:
          buf_prop_list = {property::buffer::mem_channel{5}};
          break;
        case 5:
          buf_prop_list = {property::buffer::mem_channel{6}};
          break;
        case 6:
          buf_prop_list = {property::buffer::mem_channel{7}};
          break;
        default:
          buf_prop_list = {property::buffer::mem_channel{1}};
          break;
      }  // End of switch for setting buffer property

      // **** Create device buffer **** //

      // Create kernel input buffer on device (memory bank selected by
      // mem_channel property)
      buffer<unsigned, 1> dev_buf(range<1>{vector_size}, buf_prop_list);

      // **** Write random values to device global memory **** ///

      // Submit copy operation (explicit copy from host to device)
      auto h2d_copy_e = q.submit([&](handler &h) {
        accessor mem(dev_buf, h);
        // Writing from host memory to device buffer
        h.copy(host_data_in, mem);
      });
      // Wait for copy operation to complete
      h2d_copy_e.wait();

      // ****  Submit kernel tasks **** //

      auto e = q.submit([&](handler &h) {
        accessor mem(dev_buf, h);
        // Work group Size (1 dimension)
        constexpr size_t kWGSize = 1024 * 32;
        constexpr size_t kSimdItems = 16;
        // Global range (1 dimension)
        // Global range should be evenly distributable by work group size ...
        // ... Pad with (kWGSize - 1) before dividing by kWGSize and rounding to
        // closest multiple
        size_t N = ((vector_size + kWGSize - 1) / kWGSize) * kWGSize;
        // temp_simd_wi = kSimdItems;
        // Dummy variable used in MemReadStream kernel to prevent memory access
        // from being optimized away
        unsigned dummy_var = 0;
        // Kernel to launch selected based on outer for loop control variable k
        switch (k) {
          case 0:  // kernel MemWriteStream
            h.parallel_for<MemWriteStream>(
                nd_range<1>(range<1>(N), range<1>(kWGSize)),
                [=](nd_item<1> it) [[intel::num_simd_work_items(kSimdItems),
                                     cl::reqd_work_group_size(1, 1, kWGSize)]] {
                  // Write global ID to memory
                  auto gid = it.get_global_id(0);
                  // As global range is larger than max_alloc_size, limit access
                  // from kernel to memory to size of global memory
                  if (gid < vector_size) mem[gid] = gid;
                });
            break;
          case 1:  // kernel MemReadStream
            h.parallel_for<MemReadStream>(
                nd_range<1>(range<1>(N), range<1>(kWGSize)),
                [=](nd_item<1> it) [[intel::num_simd_work_items(kSimdItems),
                                     cl::reqd_work_group_size(1, 1, kWGSize)]] {
                  // Read memory
                  auto gid = it.get_global_id(0);
                  // As global range is larger than max_alloc_size, limit access
                  // from kernel to memory to size of global memory
                  if (gid < vector_size) {
                    unsigned val = mem[gid];
                    // Use val to prevent compiler from optimizing away this
                    // variable & read from memory
                    if (val && (dummy_var == 3))
                      mem[gid] = 2;  // Randomly selected value
                  }
                });
            break;
          case 2:  // MemReadWriteStream (also the default)
          default:
            h.parallel_for<MemReadWriteStreamNDRange>(
                nd_range<1>(range<1>(N), range<1>(kWGSize)),
                [=](nd_item<1> it) [[intel::num_simd_work_items(kSimdItems),
                                     cl::reqd_work_group_size(1, 1, kWGSize)]] {
                  // Read, modify and write to memory
                  auto gid = it.get_global_id(0);
                  if (gid < vector_size) mem[gid] = mem[gid] + 2;
                });
            break;
        }  // End of switch in q.submit
      });
      // Wait for kernel tasks to complete
      e.wait();

      // **** Calculate bandwidth for each memory bank for each kernel **** //

      if (k == 0 ||
          k == 1) {  // Unidirectional (MemReadStream or MemWriteStream kernel)
        // bw_e[k][b] = (vector_size * sizeof(unsigned) / MB) /
        // (SyclGetQStExecTimeNs(e) / 1000000000.0);
        bw_bank.push_back(((vector_size * sizeof(unsigned) / kMB) /
                           (SyclGetQStExecTimeNs(e) / 1000000000.0)));
      } else {  // bidirectional (MemReadWriteStream kernel)
        bw_bank.push_back(((vector_size * sizeof(unsigned) * 2 / kMB) /
                           (SyclGetQStExecTimeNs(e) / 1000000000.0)));
      }

      // **** Read data back from device **** //

      // Submit copy operation (copy from device to host)
      auto d2h_copy_e = q.submit([&](handler &h) {
        accessor mem(dev_buf, h);
        // Reading from device buffer into host memory
        h.copy(mem, host_data_out);
      });
      d2h_copy_e.wait();

      // **** Verification **** //

      // kernel MemReadWriteStream adds 2 to the globaloffset (where global
      // range is vector_size)
      if ((k == 0) || (k == 2)) {
        unsigned val_to_add = (k == 2) ? 2 : 0;
        bool result = true;
        int prints = 0;
        for (size_t j = 0; j < vector_size; j++) {
          unsigned input_data = (k == 2) ? (host_data_in[j]) : j;
          if (host_data_out[j] != (input_data + val_to_add)) {
            if (prints++ < 512) {  // only print 512 errors
              std::cout << "Error! Mismatch at element " << j << ":" << std::hex
                        << std::showbase << host_data_out[j]
                        << " != " << (input_data + val_to_add)
                        << std::noshowbase << std::dec
                        << "\n";  // Restoring std::cout format
            }
            result = false;
          }
        }
        if (!result) {
          std::cout << "Verification failed, terminating test\n";
          return 1;
        }
      }
    }  // End of for loop controlled by num_banks

    bw_kern.push_back(bw_bank);

  }  // End of for loop controlled by kNumKernels

  // **** Report bandwidth calculation results **** //

  std::cout << "\nSummarizing bandwidth in MB/s/bank for banks 1 to "
            << num_banks << "\n";
  float avg_bw_e = 0.0;
  for (unsigned k = 0; k < kNumKernels; k++) {
    for (unsigned b = 0; b < num_banks; b++) {
      std::cout << " " << bw_kern[k][b] << " ";
      // Accumulate data from each kernel task to calculate average bandwidth
      avg_bw_e += bw_kern[k][b];
    }
    std::cout << " " << kernel_name[k] << "\n";
  }
  // Average bandwidth
  avg_bw_e /= num_banks * kNumKernels;

  // **** Read theoretical bandwidth from board_spec.xml, compare and report
  // **** //

  // Board bandwidth utilization
  // the global memory type used for analysis is the one starts at address 0

  if (xml_found) {
    // board_spec.xml found in previous loop parsing through shim directory
    if (SHIM_PATH != NULL) {
      // The SHIM_PATH variable already has board_spec.xml appended from
      // previous loop, open file
      FILE *board_sp = std::fopen(SHIM_PATH, "r");
      if (!board_sp) {
        // Failed to open board_spec.xml
        std::cout
            << "Error in opening board spec.xml. \n"
            << "Cannot read memory bank information from board_spec.xml file,"
            << "this test will run without reporting the global memory "
            << "information from board_spec.xml.\n";
      } else {
        // board_spec.xml open, parse for global memory interface tag
        char *xml_bw = (char *)malloc(100 * sizeof(char));
        float max_bandwidth;
        int num_interfaces = 0;
        bool found = false;
        bool name_found = false;
        bool mem_found = false;
        char *mem_name = (char *)malloc(100 * sizeof(char));
        char *addr = (char *)malloc(100 * sizeof(char));
        while (std::feof(board_sp) == 0) {
          char *tmp_data = (char *)malloc(100 * sizeof(char));
          char tmp_data_str[100];
          size_t len = 0;
          fscanf(board_sp, "%s", tmp_data);
          // board_spec.xml used <global_mem> tag to represent all device memory
          // infomation
          if (!mem_found && strcmp(tmp_data, "<global_mem") == 0) {
            mem_found = true;
          }
          // Loop for name of global memory (name attribute for <global_mem>
          // tag)
          else if (!name_found && mem_found &&
                   strncmp(tmp_data, "name", 4) == 0) {
            len = strlen(tmp_data);
            assert(strlen(tmp_data) < sizeof(tmp_data_str));
            strncpy(tmp_data_str, tmp_data, len + 1);
            tmp_data_str[sizeof(tmp_data_str) - 1] = '\0';
            name_found = true;
            strcpy(mem_name, (strtok(tmp_data_str, "=\"")));
            strcpy(mem_name, (strtok(NULL, "=\"")));
          }
          // Look for maximum theoretical bandwidth in board_spec.xml
          // (max_bandwidth attribute for <global_mem> tag)
          else if (mem_found && strncmp(tmp_data, "max_bandwidth", 13) == 0) {
            strcpy(xml_bw, (strtok(tmp_data, "=\"")));
            strcpy(xml_bw, (strtok(NULL, "=\"")));
            // count the number of interfaces
            while (!std::feof(board_sp)) {
              char *tmp_rd_data = (char *)malloc(100 * sizeof(char));
              fscanf(board_sp, "%s", tmp_rd_data);
              if (strncmp(tmp_rd_data, "</global_mem>", 13) == 0) {
                free(tmp_rd_data);
                break;
              }
              // Independent <interface> tag for each memory bank
              else if (strncmp(tmp_rd_data, "<interface", 10) == 0) {
                num_interfaces++;
              }
              // Starting address of the memory bank represented by address
              // attribute in <interface> tag
              else if (strncmp(tmp_rd_data, "address", 7) == 0 && !found) {
                strcpy(addr, (strtok(tmp_rd_data, "=\"")));
                strcpy(addr, (strtok(NULL, "=\"")));
                if (strncmp(addr, "0x0", 3) == 0) found = true;
              }
              free(tmp_rd_data);
            }
            if (found) {
              free(tmp_data);
              break;  // No need to continue iteration once all global memory
                      // information is found
            }
            num_interfaces = 0;
            mem_found = false;
            name_found = false;
          }
          free(tmp_data);
        }

        // Close board_spec.xml once parsing is done
        if (std::fclose(board_sp) != 0)
          std::cout << "Error closing board_spec.xml\n\n";

        // **** Report information read from board_spec.xml **** //

        if (found && num_interfaces != 0) {
          // Name of global memory
          if (name_found)
            std::cout << "\nName of the global memory type      :    "
                      << mem_name << "\n";
          else
            std::cout << "\nName of the global memory type not found in the "
                      << "board_spec.xml \n";

          // Number of memory banks/interfaces
          std::cout << "Number Of Interfaces            :    " << num_interfaces
                    << "\n";
          std::cout << "Max Bandwidth (all memory interfaces)   :    " << xml_bw
                    << " MB/s \n";
#ifndef FPGA_EMULATOR
          // Max theoretical bandwidth of global memory
          max_bandwidth = std::atof(xml_bw);

          // Theoretical bandwidth of each memory bank
          max_bandwidth = max_bandwidth / num_interfaces;

          // % utilization calculated based on average bandwidth measured by
          // this test and theoretical one A low number indicates in inefficient
          // usage of memory interface bandwidth
          float utilization = (avg_bw_e / max_bandwidth) * 100;

          std::cout
              << "Max Bandwidth of 1 memory interface in board_spec.xml :    "
              << max_bandwidth << " MB/s \n\n";

          std::cout << "It is assumed that all memory interfaces have equal "
                    << "widths. \n\n";
          std::cout << "BOARD BANDWIDTH UTILIZATION = " << utilization << "\n";

          if (utilization < 90)
            std::cout
                << "Warning : Board bandwidth utilization is less than 90% \n";
#else
          std::cout << "\nRead data from board_spec.xml, skipping utilization "
                    << "data reporting in emulation\n";
#endif
        } else {
          std::cout << "Warning: could not find global memory information in "
                    << "board spec.  \n";
        }

        free(addr);
        free(mem_name);
        free(xml_bw);
      }
    }
  } else {
    // Board_spec.xml file not found, cannot report information
    std::cout
        << "\nCannot read and report global memory information from "
        << "board_spec.xml file, only measured value from this test will be "
        << "reported.\n"
        << "\nPlease set environment variable <OFS_OCL_SHIM_ROOT_HW> to "
        << "<path-to-OpenCL/oneAPI-shim/hardware/<board_variant>/> directory.\n"
        << "If this environment variable is set, please ensure there is a "
        << "board_spec.xml file in $OFS_OCL_SHIM_ROOT_HW.\n\n";
  }

  // Report average kernel memory bandwidth
  std::cout << "\nKERNEL-TO-MEMORY BANDWIDTH = " << avg_bw_e << " MB/s/bank\n";

  // Free allocated host memory
  free(SHIM_PATH);
  free(host_data_in);
  free(host_data_out);

  return 0;
}
