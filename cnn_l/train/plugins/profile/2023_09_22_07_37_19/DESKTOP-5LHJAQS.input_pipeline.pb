	|*�=%P@|*�=%P@!|*�=%P@	�gzO���?�gzO���?!�gzO���?"{
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
	Ւ�r0;@Ւ�r0;@!Ւ�r0;@      ��!       "      ��!       *      ��!       2	��"M"N@��"M"N@!��"M"N@:      ��!       B      ��!       J	���sE)�?���sE)�?!���sE)�?R      ��!       Z	���sE)�?���sE)�?!���sE)�?b      ��!       JCPU_ONLYY�gzO���?b qf!���X@