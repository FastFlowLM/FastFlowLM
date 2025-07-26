#include "npu_utils/npu_instr_utils.hpp"

///@brief constructor
///@param npu_seq the npu sequence
///@note The function will copy the npu sequence and parse it
npu_sequence::npu_sequence(npu_device device_gen, xrt::device* device, xrt::kernel* kernel, std::string app_name){
    this->device_gen = device_gen;
    this->setup_device(device_gen);
    this->belonging_device = device;
    this->belonging_kernel = kernel;
    this->instr_name = app_name;
    this->npu_seq = nullptr;
    this->is_valid = false;
    this->clear_cmds();
}


///@brief constructor
///@param filename the filename of the npu sequence or the name of the npu sequence if it is not a file
///@param from_file default is true, if true, the function will read the npu sequence from the file; if false, the function will use the filename as the name of the npu sequence;
///@note The function will read the npu sequence from the file and parse it
///@note If the file is not found, the function will throw an error
///@warning If the from_file is false, the function will not check if the filename is valid, and the npu sequence is empty
void npu_sequence::from_file(std::string filename, bool is_binary){
    if (!this->_check_bo_valid()){
        throw std::runtime_error("NPU sequence is not binded to a device and kernel!");
    }
    std::ifstream instr_file(filename, std::ios::binary);
    if (!instr_file.is_open()){
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::vector<uint32_t> instr_v;
    if (is_binary){
        instr_file.seekg(0, std::ios::end);
        size_t instr_size = instr_file.tellg();
        instr_file.seekg(0, std::ios::beg);

        instr_v.resize(instr_size / 4);

        if (instr_size % 4 != 0){
            throw std::runtime_error("Instr file is invalied!");
        }

        if (!instr_file.read(reinterpret_cast<char *>(instr_v.data()), instr_size)) {
            throw std::runtime_error("Failed to read instruction file\n");
        }
    }
    else{
        std::string line;
        while (std::getline(instr_file, line)) {
            std::istringstream iss(line);
            uint32_t a;
            if (!(iss >> std::hex >> a)) {
                throw std::runtime_error("Unable to parse instruction file\n");
            }
            instr_v.push_back(a);
        }
    }

    if ((this->npu_seq == nullptr) || (this->npu_seq->size() != instr_v.size())){
        this->npu_seq = std::make_unique<buffer<uint32_t>>(instr_v.size(), *(this->belonging_device), *(this->belonging_kernel), 1, XRT_BO_FLAGS_CACHEABLE);
    }
    this->npu_seq->copy_from(instr_v);
    this->sync();
    this->seq2cmds();
    this->is_valid = true;
}


///@brief name the npu sequence
///@param instr_name the name of the npu sequence
///@note The function will set the name of the npu sequence
///@warning The name of the npu sequence is not used in the npu sequence, it is only for the user to identify the npu sequence
void npu_sequence::name_instr(std::string instr_name){
    this->instr_name = instr_name;
}

///@brief parse the npu sequence
///@note The function will parse the npu sequence and set the npu major, minor, dev gen, rows, cols, mem tile rows, instruction counts, instruction lines
void npu_sequence::seq2cmds(){
    if (!this->_check_bo_valid()){
        throw std::runtime_error("NPU sequence is not binded to a device and kernel!");
    }
    assert(this->npu_seq->size() > 0);
    uint32_t* seq = this->npu_seq->data();
    this->clear_cmds();
    // Parse the npu sequence
    this->npu_major = (seq[0] >> dev_major_shift) & dev_major_mask;
    this->npu_minor = (seq[0] >> dev_minor_shift) & dev_minor_mask;
    this->npu_dev_gen = (seq[0] >> dev_gen_shift) & dev_gen_mask;
    this->npu_rows = (seq[0] >> dev_n_row_shift) & dev_n_row_mask;
    this->npu_cols = (seq[1] >> dev_num_cols_shift) & dev_num_cols_mask;
    this->npu_mem_tile_rows = (seq[1] >> dev_mem_tile_rows_shift) & dev_mem_tile_rows_mask;
    this->instruction_counts = seq[2];
    this->instruction_lines = seq[3] / 4;
    int i = 4;
    while (i < this->npu_seq->size()){
        if (seq[i] == op_headers::dma_block_write){
            LOG_VERBOSE(1, "DMA block write");
            std::unique_ptr<npu_dma_block_cmd> cmd = std::make_unique<npu_dma_block_cmd>();
            cmd->dump_cmd(seq + i);
            i += cmd->get_op_lines();
            this->cmds.push_back(std::move(cmd));
            LOG_VERBOSE(1, "DMA block write" << i);
        }
        else if (seq[i] == op_headers::dma_ddr_patch_write){
            LOG_VERBOSE(1, "DMA DDR patch write");
            std::unique_ptr<npu_ddr_cmd> cmd = std::make_unique<npu_ddr_cmd>();
            cmd->dump_cmd(seq + i);
            i += cmd->get_op_lines();
            this->cmds.push_back(std::move(cmd));
        }
        else if (seq[i] == op_headers::dma_mask_write){
            LOG_VERBOSE(1, "DMA issue token write");
            std::unique_ptr<npu_issue_token_cmd> cmd = std::make_unique<npu_issue_token_cmd>();
            cmd->dump_cmd(seq + i);
            i += cmd->get_op_lines();
            this->cmds.push_back(std::move(cmd));
        }
        else if (seq[i] == op_headers::queue_write){
            LOG_VERBOSE(1, "Queue write");
            std::unique_ptr<npu_write_cmd> cmd = std::make_unique<npu_write_cmd>();
            cmd->dump_cmd(seq + i);
            i += cmd->get_op_lines();
            this->cmds.push_back(std::move(cmd));
        }
        else if (seq[i] == op_headers::dma_sync_write){ // Wait sync, AIETargetNPU.cpp line 62
            LOG_VERBOSE(1, "DMA sync write");
            std::unique_ptr<npu_wait_cmd> cmd = std::make_unique<npu_wait_cmd>();
            cmd->dump_cmd(seq + i);
            i += cmd->get_op_lines();
            this->cmds.push_back(std::move(cmd));
        }
        else{
            i++;
        }
    }
}

///@brief print the npu sequence
///@note The function will print the npu major, minor, dev gen, rows, cols, mem tile rows, instruction counts, instruction lines and the commands in the npu sequence
void npu_sequence::interpret(){
    int line_number = 0;
    MSG_BONDLINE(INSTR_PRINT_WIDTH);
    uint32_t* seq = this->npu_seq->data();
    instr_print(line_number, seq[line_number], "NPU information");
    instr_print(-1, seq[line_number], "--NPU version: " + std::to_string(this->npu_major) + "." + std::to_string(this->npu_minor));
    instr_print(-1, seq[line_number], "--NPU generation: " + std::to_string(this->npu_dev_gen));
    instr_print(-1, seq[line_number], "--NPU rows: " + std::to_string(this->npu_rows));
    line_number++;
    instr_print(line_number, seq[line_number], "--NPU cols: " + std::to_string(this->npu_cols));
    instr_print(-1, seq[line_number], "--NPU memory tile rows: " + std::to_string(this->npu_mem_tile_rows));
    line_number++;

    // Instruction commands
    instr_print(line_number, seq[line_number], "Instruction commands: " + std::to_string(seq[line_number]));
    line_number++;

    // Instruction lines
    instr_print(line_number, seq[line_number], "Instruction lines: " + std::to_string(seq[line_number] / 4));
    line_number++;

    for (int i = 0; i < this->cmds.size(); i++){
        line_number = this->cmds[i]->print_cmd(seq + line_number, line_number, i);
    }
    MSG_BONDLINE(INSTR_PRINT_WIDTH);
}

///@brief convert the npu sequence to the npu format
///@note The function will convert the npu sequence to the npu format
///@note If the npu sequence is pre-generated(created from a file or other vector), the function will compare the npu sequence with the pre-generated npu sequence and print the difference
///@note If the npu sequence is not pre-generated, the function will update the npu sequence inside the class
///@warning If a npu sequence is not pre-generated, this function must be called before the npu sequence is used by the npu_app
///@return the npu sequence in std::vector<uint32_t>
void npu_sequence::cmds2seq(){
    std::vector<uint32_t> npu_seq_generated;
    
    npu_seq_generated.push_back(
        (this->npu_major << dev_major_shift) |
        (this->npu_minor << dev_minor_shift) |
        (this->npu_dev_gen << dev_gen_shift) |
        (this->npu_rows << dev_n_row_shift)
    );
    
    npu_seq_generated.push_back(
        (this->npu_cols << dev_num_cols_shift) |
        (this->npu_mem_tile_rows << dev_mem_tile_rows_shift)
    );

    this->instruction_counts = this->cmds.size();
    this->instruction_lines = 4;
    for (int i = 0; i < this->cmds.size(); i++){
        this->instruction_lines += this->cmds[i]->get_op_lines();
    }
    npu_seq_generated.push_back(this->instruction_counts);
    npu_seq_generated.push_back(this->instruction_lines << 2);
    for (int i = 0; i < this->cmds.size(); i++){
        this->cmds[i]->to_npu(npu_seq_generated);
    }

    if ((this->npu_seq == nullptr) || (this->npu_seq->size() != npu_seq_generated.size())){
        this->npu_seq = std::make_unique<buffer<uint32_t>>(npu_seq_generated.size(), *this->belonging_device, *this->belonging_kernel, 1, XRT_BO_FLAGS_CACHEABLE);
    }

    this->npu_seq->copy_from(npu_seq_generated);
    this->sync();
    this->is_valid = true;
}


void npu_sequence::write_out_sequence(std::string filename){
    assert(this->is_valid);
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()){
        throw std::runtime_error("Failed to open file: " + filename);
    }

    file.write(reinterpret_cast<const char*>(this->npu_seq->data()), this->npu_seq->size() * sizeof(uint32_t));
    file.close();
}

buffer<uint32_t> npu_sequence::dump(){
    buffer<uint32_t> copy_buf = buffer<uint32_t>(this->npu_seq->size());
    copy_buf.copy_from(*(this->npu_seq));
    return copy_buf;
}

///@brief setup the npu device
///@param device the npu device
///@note The function will setup the npu device
///@warning The function is only used for the npu sequence that is not pre-generated
void npu_sequence::setup_device(npu_device device){
    if (device == device_npu1){
        // Might be wrong
        this->npu_major = 0;
        this->npu_minor = 1;
        this->npu_dev_gen = 1;
        this->npu_rows = 6;
        this->npu_cols = 4;
        this->npu_mem_tile_rows = 1;
    }
    else if (device == device_npu2){
        this->npu_major = 0;
        this->npu_minor = 1;
        this->npu_dev_gen = 4;
        this->npu_rows = 6;
        this->npu_cols = 8;
        this->npu_mem_tile_rows = 1;
    }
}

///@{
/**
 *  @brief generate a npu write command  
 *  @param tile: the tile of the NPU
 *  @param addr: the address of the register
 *  @param value: the value to write to the register
 */
void npu_sequence::rtp_write(npu_tiles tile, uint32_t addr, uint32_t value){
    std::unique_ptr<npu_write_cmd> cmd = std::make_unique<npu_write_cmd>();
    uint32_t row = (tile >> 4) & 0xF;
    uint32_t col = tile & 0xF;
    cmd->row = row;
    cmd->col = col;
    cmd->reg_addr = addr;
    cmd->value = value;
    cmd->could_be_push_queue = false;
    this->is_valid = false;
    this->cmds.push_back(std::move(cmd));
}


///@{
/**
 *  @brief generate a npu dma memory copy command  
 *  @param elem_size: the size of the element in bytes
 *  @param arg_idx: the index of the argument in the kernel
 *  @param channel_direction: the direction of the channel
 *  @param tile: the tile of the NPU
 *  @param bd_id: the BD ID of the NPU
 *  @param it_channel: the IT channel of the NPU
 *  @param offset: the offset of the memory in the DDR
 *  @param size: the size of the memory in the DDR
 *  @param stride: the stride of the memory in the DDR
 *  @param packet_id: the ID of the packet, -1 is disable it
 *  @param packet_type: the type of the packet, mostly 0
 *  @param issue_token: whether to issue a token, MM2S is false, S2MM is true by default
 *  @return void
 */
void npu_sequence::npu_dma_memcpy_nd(
    int elem_size,
    int arg_idx,
    dma_direction channel_direction,
    npu_tiles tile,
    npu_bd_id bd_id,
    npu_it_channel it_channel,
    const std::vector<uint32_t>& _offset,
    const std::vector<uint32_t>& _size,
    const std::vector<uint32_t>& _stride,
    int packet_id,
    int packet_type,
    bool issue_token
){
    assert(elem_size <= 4); // not supported
    this->is_valid = false;
    std::vector<uint32_t> offset = _offset;
    std::vector<uint32_t> size = _size;
    std::vector<uint32_t> stride = _stride;
    uint32_t row = (tile >> 4) & 0xF;
    uint32_t col = tile & 0xF;
    if (channel_direction == S2MM){
        issue_token = true;
    }

    if (elem_size == 1){
        // 1 byte data, AIE convert it to uint32_t
        elem_size = 4;
        size[3] >>= 2;
        offset[3] >>= 2;
        for (int i = 2; i >= 0; i--){
            stride[i] >>= 2;
        }
    }
    else if (elem_size == 2){
        // 2 byte data, AIE convert it to uint32_t
        elem_size = 4;
        size[3] >>= 1;
        offset[3] >>= 1;
        for (int i = 2; i >= 0; i--){
            stride[i] >>= 1;
        }
    }
    std::unique_ptr<npu_dma_block_cmd> cmd = std::make_unique<npu_dma_block_cmd>();
    cmd->row = row;
    cmd->col = col;
    cmd->bd_id = bd_id;
    if (size[1] == 1 && size[2] == 1){
        cmd->is_linear = true;
    }
    else{
        cmd->is_linear = false;
    }
    // reverse the order of offset, size and stride
    cmd->buffer_offset = 0;
    cmd->buffer_length = size[3];
    for (int i = 2; i >= 1; i--){ // iteration size does not count
        cmd->buffer_length *= size[i];
    }
    
    cmd->packet_enable = packet_id != -1;
    cmd->out_of_order_id = 0;
    if (cmd->packet_enable){
        cmd->packet_id = packet_id;
        cmd->packet_type = packet_type;
    }
    else{
        cmd->packet_id = 0;
        cmd->packet_type = 0;
    }
    
    if (cmd->is_linear){
        cmd->dim0_size = 0;
        cmd->dim0_stride = 1;
        cmd->dim1_size = 0;
        cmd->dim1_stride = 1;
        cmd->dim2_size = 0;
        cmd->dim2_stride = 1;
    }
    else{
        // inverse of human's order
        cmd->dim0_size = size[3];
        cmd->dim0_stride = size[3] != 1 ? stride[3] : 1;
        cmd->dim1_size = size[2];
        cmd->dim1_stride = size[2] != 1 ? stride[2] : 1;
        cmd->dim2_size = size[1];
        cmd->dim2_stride = size[1] != 1 ? stride[1] : 1;
    }
    cmd->next_bd_id = 0;
    cmd->valid_bd = 1;
    if (cmd->is_linear){
        cmd->iter_size = 1;
        cmd->iter_stride = 1;
    }
    else{
        cmd->iter_size = size[0];
        if (cmd->iter_size > 1){
            cmd->iter_stride = stride[0];
        }
        else{
            cmd->iter_stride = 1;
        }
    }
    cmd->issue_token = issue_token;

    cmd->get_lock_rel_val = 0;
    cmd->get_lock_rel_id = 0;
    cmd->get_lock_acq_enable = 0;
    cmd->get_lock_acq_val = 0;
    cmd->get_lock_acq_id = 0;

    this->cmds.push_back(std::move(cmd));
    // add ddr patch

    std::unique_ptr<npu_ddr_cmd> ddr_cmd = std::make_unique<npu_ddr_cmd>();
    ddr_cmd->row = row;
    ddr_cmd->col = col;
    ddr_cmd->bd_id = bd_id;
    ddr_cmd->arg_offset = offset[3];
    for (int i = 2; i >= 0; i--){
        ddr_cmd->arg_offset += offset[i] * stride[i];
    }
    ddr_cmd->arg_offset *= elem_size;
    ddr_cmd->arg_idx = arg_idx;

    // add issue token
    if (issue_token){
        std::unique_ptr<npu_issue_token_cmd> issue_token_cmd = std::make_unique<npu_issue_token_cmd>();
        issue_token_cmd->row = row;
        issue_token_cmd->col = col;
        issue_token_cmd->channel_direction = channel_direction;
        issue_token_cmd->channel_id = it_channel;
        issue_token_cmd->controller_packet_id = 15;
        this->cmds.push_back(std::move(issue_token_cmd));
    }
    
    this->cmds.push_back(std::move(ddr_cmd));
    // queue write
    std::unique_ptr<npu_write_cmd> queue_cmd = std::make_unique<npu_write_cmd>();
    queue_cmd->row = row;
    queue_cmd->col = col;
    queue_cmd->channel_direction = channel_direction;
    queue_cmd->could_be_push_queue = true;
    queue_cmd->channel_id = it_channel;
    queue_cmd->repeat_count = size[0] - 1;
    queue_cmd->issue_token = issue_token;
    queue_cmd->bd_id = bd_id;
    this->cmds.push_back(std::move(queue_cmd));
}

///@{
/**
 *  @brief generate a npu wait command  
 *  @param tile: the tile of the NPU
 *  @param channel_direction: the direction of the channel
 *  @param it_channel: the IT channel of the NPU
 *  @return void
 */
void npu_sequence::npu_dma_wait(npu_tiles tile, dma_direction channel_direction, npu_it_channel it_channel){
    this->is_valid = false;
    std::unique_ptr<npu_wait_cmd> wait_cmd = std::make_unique<npu_wait_cmd>();
    uint32_t row = (tile >> 4) & 0xF;
    uint32_t col = tile & 0xF;
    wait_cmd->wait_row = row;
    wait_cmd->wait_col = col;
    wait_cmd->channel_direction = channel_direction;
    wait_cmd->wait_channel = it_channel;
    this->cmds.push_back(std::move(wait_cmd));
}

///@{
/**
 *  @brief generate a npu maskwrite command  
 *  @param tile: the tile of the NPU
 *  @param addr: the address of the register
 *  @param value: the value to write to the register
 *  @param mask: the mask of the register
 *  @return void
 */
void npu_sequence::npu_maskwrite(npu_tiles tile, uint32_t addr, uint32_t value, uint32_t mask){
    this->is_valid = false;
    std::unique_ptr<npu_maskwrite_cmd> maskwrite_cmd = std::make_unique<npu_maskwrite_cmd>();
    uint32_t row = (tile >> 4) & 0xF;
    uint32_t col = tile & 0xF;
    maskwrite_cmd->row = row;
    maskwrite_cmd->col = col;
    maskwrite_cmd->addr = addr;
    maskwrite_cmd->value = value;
    maskwrite_cmd->mask = mask;
    this->cmds.push_back(std::move(maskwrite_cmd));
}

xrt::bo& npu_sequence::bo(){
    assert(this->is_valid);
    assert(this->npu_seq != nullptr);
    return this->npu_seq->bo();
}

void npu_sequence::clear_cmds(){
    this->cmds.clear();
    this->is_valid = false;
}