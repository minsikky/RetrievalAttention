import time
import os
import resource
import subprocess
import torch
from termcolor import colored


class LLM:
    """
    A class representing the LLM (currently support Llama and Qwen).
    """

    def __init__(
        self, 
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
        device_map: str
    ) -> None:
        """ Initializes the LLM.
        Args:
            model_name (str): The name of the model.
            max_length (int): The maximum length (prefill+decode) of sequences.
            dtype (torch.dtype): The data type for model computations.
            device_map (str): The device for model, suppor 'cuda:x' or 'auto (automatically use all visible GPUs)'.
        """

        self.model_name = model_name
        self.max_length = max_length
        self.dtype = dtype
        self.device_map = device_map

    def _get_rss_mb(self):
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 ** 2)
        except Exception:
            try:
                with open("/proc/self/status", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            parts = line.split()
                            return float(parts[1]) / 1024.0  # kB -> MB
            except Exception:
                usage = resource.getrusage(resource.RUSAGE_SELF)
                return float(usage.ru_maxrss) / 1024.0  # kB -> MB on Linux
        return None

    def _get_gpu_mem_mb(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            values = [v.strip() for v in result.stdout.splitlines() if v.strip()]
            return ",".join(values) if values else None
        except Exception:
            return None

    def _log_mem(self, tag):
        rss = self._get_rss_mb()
        gpu = self._get_gpu_mem_mb()
        rss_str = f"{rss:.1f} MB" if rss is not None else "N/A"
        gpu_str = f"{gpu} MB" if gpu is not None else "N/A"
        try:
            cuda_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            cuda_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            cuda_str = f"{cuda_alloc:.1f}/{cuda_reserved:.1f} MB"
        except Exception:
            cuda_str = "N/A"
        print(f"[MEM] {tag} | RSS={rss_str} | GPU={gpu_str} | CUDA(alloc/res)={cuda_str}")


    def layer_prefill(self, layer_idx, start_bdx, hidden_states):
        # print(f'Layer = {layer_idx}, start_bdx = {start_bdx}')

        bsz, seq_len, dim = hidden_states.shape
        layer = self.layers[layer_idx]
        
        # original hidden_states used as residual, clone a new one to process
        temp_hidden_states = hidden_states.clone()

        # chunk for lower memory comsumption
        for start_idx in range(0, seq_len, 8192//bsz):
            end_idx = min(seq_len, start_idx + 8192//bsz)
            temp_hidden_states[:, start_idx:end_idx, :] = self.layernorm(temp_hidden_states[:, start_idx:end_idx, :], 
                                                                         layer.input_layernorm_variance_epsilon, 
                                                                         layer.input_layernorm_weight)
        
        query_states, key_states, value_states = self.wqkv(temp_hidden_states, layer)
        del temp_hidden_states
        torch.cuda.empty_cache()
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)       # reshape [bs, seq_len, dim] => [bs, seq_len, head, head_dim]
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        key_states, value_states = self.kv_cache.prefill_update_kv_cache(query_states, key_states, value_states, layer_idx, start_bdx)
        torch.cuda.empty_cache()

        temp_attn_out = self.prefill_attention(
            query_states,
            key_states,
            value_states,
            layer_idx=layer_idx,
        )

        self.kv_cache.sync(layer_idx, start_bdx)

        del query_states, key_states, value_states
        torch.cuda.empty_cache()

        hidden_states += self.wo(temp_attn_out, layer, temp_attn_out.shape[0], seq_len, dim)
        del temp_attn_out
        torch.cuda.empty_cache()

        # post attention
        residual = hidden_states.clone()

        # chunk for lower memory comsumption
        for start_idx in range(0, seq_len, 8192//bsz):
            end_idx = min(seq_len, start_idx + 8192//bsz)
            hidden_states[:, start_idx:end_idx, :] = self.layernorm(hidden_states[:, start_idx:end_idx, :], 
                                                                    layer.post_attention_layernorm_variance_epsilon, 
                                                                    layer.post_attention_layernorm_weight)
            hidden_states[:, start_idx:end_idx, :] = self.mlp(hidden_states[:, start_idx:end_idx, :], layer)   
        
        hidden_states += residual

        del residual
        torch.cuda.empty_cache()
                                                                                                   
        return hidden_states


    def layer_decode(self, layer_idx, hidden_states):
        # print(f'Layer = {layer_idx}')

        residual = hidden_states
        bsz, seq_len, dim = hidden_states.shape
        layer = self.layers[layer_idx]

        hidden_states = self.layernorm(hidden_states, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        
        query_states, key_states, value_states = self.wqkv(hidden_states, layer)
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim)

        key_states, value_states = self.kv_cache.decode_update_kv_cache(key_states, value_states, layer_idx)
        attn_out = self.decode_attention(query_states, key_states, value_states, layer_idx)
        hidden_states = self.wo(attn_out, layer, bsz, seq_len, dim)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layernorm(hidden_states, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
        hidden_states = self.mlp(hidden_states, layer)
        hidden_states = residual + hidden_states

        return hidden_states


    def prefill_forward(self, inputs_ids):
        bsz, seq_len = inputs_ids.shape
        device = inputs_ids.device

        last_hidden_states = torch.empty((bsz, 1, self.hidden_size), dtype=self.dtype, device=device)
        for start_bdx in range(0, bsz, 1):
            end_bdx = min(bsz, start_bdx + 1)
            hidden_states = self.word_embedding(inputs_ids[start_bdx:end_bdx])  # [1, seq_len, hidden_size]

            if self.num_gpus > 1:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                    hidden_states = self.parameter_move(hidden_states, ldx)
                    torch.cuda.empty_cache()
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :].to(self.layers[0].device)
            else:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                    torch.cuda.empty_cache()
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :]
        
        last_hidden_states = self.layernorm(last_hidden_states.contiguous(), self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(last_hidden_states)
        
        return logits
        

    def decode_forward(self, inputs_ids):
        hidden_states = self.word_embedding(inputs_ids)

        if self.num_gpus > 1:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states)
                hidden_states = self.parameter_move(hidden_states, ldx)
            hidden_states = hidden_states.to(self.layers[0].device)
        else:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states)
        
        hidden_states = self.layernorm(hidden_states[:, -1:, :], self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(hidden_states)
        
        return logits


    def inference(self, inputs_ids):
        outputs_ids = []    # multi iteration, multi request
        output_ids = []     # single iteration, multi request
        decode_steps = max(0, self.max_new_length - 1)
        
        print("Start prefilling ...")
        self._log_mem("prefill.start")
        torch.cuda.synchronize()
        prefill_start = time.time()

        enable_profiler = (
            os.getenv("ENABLE_PROFILER", "0") == "1"
            and os.getenv("PROFILER_SAFE", "0") == "1"
        )
        profiler_dir = os.getenv("PROFILER_DIR", "profiling")
        if enable_profiler:
            os.makedirs(profiler_dir, exist_ok=True)
            try:
                from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
                trace_dir = os.path.join(profiler_dir, f"profile_{int(time.time())}")
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    schedule=schedule(wait=1, warmup=1, active=2, repeat=0),
                    profile_memory=True,
                    record_shapes=False,
                    on_trace_ready=tensorboard_trace_handler(trace_dir),
                ) as prof:
                    logits = self.prefill_forward(inputs_ids=inputs_ids)
                    prof.step()
                    output_ids = logits.argmax(dim=-1)
                    outputs_ids.append(output_ids)
                    self.move()

                    torch.cuda.synchronize()
                    prefill_end = time.time()
                    print(colored(f"Prefilling latency: {round((prefill_end - prefill_start), 4)} s\n", 'green'))
                    self._log_mem("prefill.end")

                    print("Start decoding ...")
                    self._log_mem("decode.start")
                    decode_start = time.time()

                    if decode_steps > 0:
                        for _ in range(decode_steps):
                            logits = self.decode_forward(inputs_ids=output_ids)
                            output_ids = logits.argmax(dim=-1)
                            outputs_ids.append(output_ids)
                            prof.step()

                        decode_end = time.time()
                        decode_total = decode_end - decode_start
                        decode_total_safe = max(decode_total, 1e-9)
                        print(colored(
                            f"Decoding total latency: {round(decode_total, 4)} s, "
                            f"Decoding latency: {round(decode_total * 1000 / decode_steps, 2)} ms/step, "
                            f"Throughput: {round(self.batch_size * decode_steps / decode_total_safe, 2)} tokens/s\n",
                            'green'
                        ))
                        if hasattr(self, "kv_cache") and hasattr(self.kv_cache, "report_decode_profile"):
                            decode_profile_msg = self.kv_cache.report_decode_profile(reset=True)
                            if decode_profile_msg:
                                print(decode_profile_msg)
                    else:
                        print(colored("Decoding skipped (max_new_length <= 1)\n", 'green'))
                    self._log_mem("decode.end")
            except Exception as e:
                print(f"[WARN] Profiler disabled due to error: {e}")
                enable_profiler = False
        else:
            logits = self.prefill_forward(inputs_ids=inputs_ids)
            output_ids = logits.argmax(dim=-1)
            outputs_ids.append(output_ids)
            self.move()

            torch.cuda.synchronize()
            prefill_end = time.time()
            print(colored(f"Prefilling latency: {round((prefill_end - prefill_start), 4)} s\n", 'green'))
            self._log_mem("prefill.end")

            print("Start decoding ...")
            self._log_mem("decode.start")
            decode_start = time.time()

            if decode_steps > 0:
                for _ in range(decode_steps):
                    logits = self.decode_forward(inputs_ids=output_ids)
                    output_ids = logits.argmax(dim=-1)
                    outputs_ids.append(output_ids)

                decode_end = time.time()
                decode_total = decode_end - decode_start
                decode_total_safe = max(decode_total, 1e-9)
                print(colored(
                    f"Decoding total latency: {round(decode_total, 4)} s, "
                    f"Decoding latency: {round(decode_total * 1000 / decode_steps, 2)} ms/step, "
                    f"Throughput: {round(self.batch_size * decode_steps / decode_total_safe, 2)} tokens/s\n",
                    'green'
                ))
                if hasattr(self, "kv_cache") and hasattr(self.kv_cache, "report_decode_profile"):
                    decode_profile_msg = self.kv_cache.report_decode_profile(reset=True)
                    if decode_profile_msg:
                        print(decode_profile_msg)
            else:
                print(colored("Decoding skipped (max_new_length <= 1)\n", 'green'))
            self._log_mem("decode.end")
        
        outputs_ids = torch.cat(outputs_ids, dim=-1).tolist()
        
        return outputs_ids


    def generate(self, attention_type, inputs_ids, attention_masks, max_new_length, attn_config=None):
        """ LLM Inference.
        Args:
            attention_type: str,
            input_ids (torch.tensor): The input of LLM.
            attention_masks (torch.tensor): The attention masks of LLM.
            max_new_length (int): The maximum length of generated sequences.
        """

        bs, input_length = inputs_ids.shape
        assert input_length + max_new_length <= self.max_length, \
        f"Error: input_length({input_length}) + max_new_length({max_new_length}) exceeds max_length({self.max_length})"

        self.batch_size = bs
        self.input_length = input_length
        self.max_new_length = max_new_length
        self.attention_type = attention_type

        valid_start = attention_masks.shape[1] - torch.sum(attention_masks, dim=-1).detach().cpu().numpy()
        del attention_masks
        torch.cuda.empty_cache()

        print("Allocate GPU buffers and CPU pin memory ...\n")
        self._log_mem("generate.before_kv_cache")
        self.init_kv_cache(input_length, valid_start, attn_config)
        self._log_mem("generate.after_kv_cache")

        outputs = self.inference(inputs_ids)

        return outputs
