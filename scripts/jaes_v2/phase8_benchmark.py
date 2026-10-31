"""Tiny real-data steady-state CPU resource benchmark, not a quality experiment."""
from pathlib import Path
from time import perf_counter
import os,tempfile,json,statistics
import numpy as np
from src.lava.jaes_v2.evidence import write_json
from src.lava.jaes_v2.phase8_training import BASE,load_config,dataset,initialize,Inputs,model_factory,variable_hash,save_checkpoint
from src.lava.jaes_v2.phase8_objectives import weighted_sum

def main():
    root=Path(__file__).resolve().parents[2];temp=root/'.research_tmp';os.environ['TEMP']=str(temp);os.environ['TMP']=str(temp);tempfile.tempdir=str(temp)
    from scripts.jaes_v2.phase6_experiment import resource_lock
    from src.lava.jaes_v2.c2_student import peak_process_mib
    with resource_lock(root):
        out=root/BASE/'benchmark/partial_cpu_v1';out.mkdir(parents=True,exist_ok=False)
        c=load_config(root,root/'configs/jaes_v2/phase8/p8_tiny_erm_partial_s6107_v3.json')
        rows,_,_,_=dataset(root,c);chosen=rows[:8];tf=initialize('cpu');full,backbone,forward=model_factory(tf,c)
        inputs=Inputs(root,c);t=perf_counter();x=[inputs.get(r,'clean')[None] for r in chosen];prepare=perf_counter()-t
        variables=full.trainable_weights;optimizer=tf.keras.optimizers.Adam(.0001);optimizer.build(variables)
        bnvars=[v for l in backbone.layers if isinstance(l,tf.keras.layers.BatchNormalization) for v in l.weights];bn_before=variable_hash(bnvars)
        @tf.function(reduce_retracing=True)
        def gradient(x,y,counter):
            with tf.GradientTape() as tape:loss=weighted_sum(y,forward(x,True,counter),[1/len(chosen)])
            gs=tape.gradient(loss,variables)
            for g in gs:tf.debugging.assert_all_finite(g,'Benchmark gradient')
            return gs
        times=[]
        for repetition in range(6):
            t=perf_counter();acc=[tf.zeros_like(v) for v in variables]
            for i,(r,value) in enumerate(zip(chosen,x)):
                gs=gradient(value,tf.constant([r.label],tf.float32),tf.constant(repetition*8+i,tf.int32))
                acc=[a+g for a,g in zip(acc,gs)]
            optimizer.apply_gradients(zip(acc,variables));times.append(perf_counter()-t)
        infer=tf.function(lambda value:full(value,training=False));forward_times=[]
        for _ in range(6):
            t=perf_counter();infer(x[0]).numpy();forward_times.append(perf_counter()-t)
        t=perf_counter();save_checkpoint(tf,full,optimizer,out/'checkpoint',dict(contract={'scope':'RESOURCE_BENCHMARK_ONLY'},step=6));save_seconds=perf_counter()-t
        if variable_hash(bnvars)!=bn_before:raise ValueError('BN changed')
        write_json(out/'measurements.json',dict(scope='RESOURCE_BENCHMARK_ONLY; no quality comparison',runtime='CPU',microbatch=1,effective_batch=8,
            sources=[r.sample_id for r in chosen],preparation_seconds=prepare,input_components=dict(inputs.timings),warmup_update_seconds=times[0],steady_update_seconds=times[1:],
            steady_update_median=statistics.median(times[1:]),forward_warmup_seconds=forward_times[0],steady_forward_median=statistics.median(forward_times[1:]),
            checkpoint_save_seconds=save_seconds,checkpoint_bytes=sum(p.stat().st_size for p in (out/'checkpoint').rglob('*') if p.is_file()),peak_process_mib=peak_process_mib(),
            total_parameters=full.count_params(),trainable_parameters=sum(int(np.prod(v.shape)) for v in variables),BN_unchanged=True,GPU_memory=None))
        from src.lava.jaes_v2.phase8_training import seal_tree
        seal_tree(out);print((out/'measurements.json').read_text())

if __name__=='__main__':main()
