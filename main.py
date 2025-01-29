import time

from HandMadeMain import bit_of_handmade_run_with_profiler, run_knn
from sklearn_main import sklearn_run_with_profiler, sklearn_main
from BERT import bert_run_with_profiler, run_BERT
from RoBERTa import roberta_run_with_profiler, run_RoBERTa


if __name__ == '__main__':
    start_time = time.time()
    # sklearn_run_with_profiler()
    bit_of_handmade_run_with_profiler()
    # roberta_run_with_profiler()
    # bert_run_with_profiler()

    print("--- Total duration: %s seconds ---" % (time.time() - start_time))
















