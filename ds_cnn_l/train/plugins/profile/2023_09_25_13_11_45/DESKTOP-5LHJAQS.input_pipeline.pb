	���do@���do@!���do@	౏l��G@౏l��G@!౏l��G@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:���do@׆�q��@A:#J{�x_@YGV~��]@rEagerKernelExecute 0*	�C���-A2�
WIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::ForeverRepeat::ParallelMapV2&*�t��qv@!��S�R@)*�t��qv@1��S�R@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchc��K�]@!C�Eۥ�8@)c��K�]@1C�Eۥ�8@:Preprocessing2
HIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::ForeverRepeat&���vv@!v�v"	�R@)s����?1#�Og��?:Preprocessing2�
dIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::ForeverRepeat::ParallelMapV2::TensorSlice&KO�\�?!c��2�?)KO�\�?1c��2�?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism[]N	��]@!*	�]1�8@)�_ѭׄ?1N���Hpa?:Preprocessing2F
Iterator::Model9�	�ʞ]@!��]i�8@)zpw�n�p?1K��f��K?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 47.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9߱�l��G@I!Np�OgJ@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	׆�q��@׆�q��@!׆�q��@      ��!       "      ��!       *      ��!       2	:#J{�x_@:#J{�x_@!:#J{�x_@:      ��!       B      ��!       J	GV~��]@GV~��]@!GV~��]@R      ��!       Z	GV~��]@GV~��]@!GV~��]@b      ��!       JCPU_ONLYY߱�l��G@b q!Np�OgJ@