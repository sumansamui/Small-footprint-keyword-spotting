�	|*�=%P@|*�=%P@!|*�=%P@	�gzO���?�gzO���?!�gzO���?"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:|*�=%P@Ւ�r0;@A��"M"N@Y���sE)�?rEagerKernelExecute 0*	���x�bA2�
WIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::ForeverRepeat::ParallelMapV2I�v|�a@!�Tsd�X@)I�v|�a@1�Tsd�X@:Preprocessing2
HIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::ForeverRepeat�>�Q�a@!��Nȯ�X@)5���#�?1��n֧Z�?:Preprocessing2�
dIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::ForeverRepeat::ParallelMapV2::TensorSliced�w��?!�X����?)d�w��?1�X����?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismƦ�B ��?!�"$�D�?);��]؊?1{i�aق?:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchQ3���U�?!����]?)Q3���U�?1����]?:Preprocessing2F
Iterator::Model&s,��?!~{��W��?)�b�dU�k?1���� Rc?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 5.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�gzO���?If!���X@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Ւ�r0;@Ւ�r0;@!Ւ�r0;@      ��!       "      ��!       *      ��!       2	��"M"N@��"M"N@!��"M"N@:      ��!       B      ��!       J	���sE)�?���sE)�?!���sE)�?R      ��!       Z	���sE)�?���sE)�?!���sE)�?b      ��!       JCPU_ONLYY�gzO���?b qf!���X@Y      Y@q5���4'@"�	
both�Your program is POTENTIALLY input-bound because 5.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb�11.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 