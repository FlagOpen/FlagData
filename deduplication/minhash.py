import hashlib
import re
import struct
import sys
from logging import Logger
from typing import Tuple, Optional
import numpy as np
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, IntegerType
from pyspark.sql import functions as F
from scipy.integrate import quad as integrate
import ftfy
from itertools import tee
from typing import List
from typing import Text
from transformers import AutoTokenizer

SEED = 42
# NON_ALPHA = re.compile("[^A-Za-z_0-9]")
# max_encode_length = 1024
RNG = np.random.RandomState(SEED)
MAX_HASH = np.uint64((1 << 32) - 1)
MERSENNE_PRIME = np.uint64((1 << 61) - 1)


def get_ngrams(sequence: List[Text], n: int, min_length: int = 5, chunk_length: Optional[int] = None):
    """
    Return the ngrams generated from a sequence of items, as an iterator. This is a modified version of nltk.util.ngrams.

    Parameters
    ----------
    sequence : List[Text]
        The sequence of items.
    n : int
        The length of each ngram.
    min_length : int, optional
        The minimum length of each ngram, by default 5

    Returns
    -------
    iterator
        The ngrams.

    Examples
    --------
    >>> list(ngrams(["a", "b", "c", "d"], 2, min_length=1))
    [('a', 'b'), ('b', 'c'), ('c', 'd')]
    >>> list(ngrams(["a", "b", "c", "d"], 2, min_length=5))
    []
    >>> list(ngrams(["a", "b"], 3, min_length=1))
    [('a', 'b')]
    """
    if len(sequence) < min_length:
        return []
    if chunk_length is not None:
        sequence = sequence[:chunk_length]
    if len(sequence) < n:
        return [tuple(sequence)]
    iterables = tee(iter(sequence), n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


def bc_tokenize(seq):
    return bc_tokenizer.value.encode(seq)


# Connected Components in MapReduce and Beyond
def large_star_map(edge):
    return [(edge[0], edge[1]), (edge[1], edge[0])]


def large_star_reduce(group):
    x, neighbors = group
    nodes = [x] + list(neighbors)
    minimum = min(nodes)
    return [(n, minimum) for n in nodes if n > x]


def small_star_map(edge):
    x, y = edge
    if y <= x:
        return (x, y)
    else:
        return (y, x)


def small_star_reduce(group):
    x, neighbors = group
    nodes = [x] + list(neighbors)
    minimum = min(nodes)
    return [(n, minimum) for n in nodes if n != minimum]


def sha1_hash32(data):
    """
    Directly taken from datasketch package to avoid dependency.

    Parameters
    ----------
    data : bytes

    Returns
    -------
    int
        The first 4 bytes (32 bits) of the SHA1 hash of the input data.

    Examples
    --------
    >>> sha1_hash32(b"hello")
    499578026
    >>> bin(sha1_hash32(b"hello"))
    '0b11101110001101111010010101010'
    >>> sha1_hash32(b"hello world").bit_length()
    30
    """
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]


"""
--threshold 参数生效第1步：
使用`generate_hash_values()`函数为给定文档生成MinHashLSH值。这些值用于衡量文档之间的相似性。
"""


def generate_hash_values(
        tokens: list,
        idx: int,
        num_perm: int,
        ngram_size: int,
        min_length: int,
        hashranges: List[Tuple[int, int]],
        permutations: np.ndarray,
) -> List[Tuple[int, bytes, int]]:
    """
    Generate the MinHashLSH values for a given document.

    Parameters
    ----------
    tokens : list
        The token ids of the document.
    idx : int
        The index of the document.
    num_perm : int
        The number of permutations.
    ngram_size : int
        The size of the n-grams.
    min_length : int
        The minimum number of tokens in a document.
    hashranges : list
        The ranges of offsets for each hash value.
    permutations : np.ndarray
        The permutations for the hash values.

    Returns
    -------
    List[Tuple[int, bytes, int]]
        The list of (band_idx, hash value, idx) for the document.

    Examples
    --------
    >>> content = "hello world"
    >>> idx = 0
    >>> num_perm = 250
    >>> ngram_size = 1
    >>> hashranges = [(i, i + 25) for i in range(0, 250, 25)]
    >>> PERMUTATIONS = np.array(
    ...     [
    ...         (
    ...             RNG.randint(1, MERSENNE_PRIME, dtype=np.uint64),
    ...             RNG.randint(0, MERSENNE_PRIME, dtype=np.uint64),
    ...         )
    ...         for _ in range(num_perm)
    ...     ],
    ...     dtype=np.uint64,
    ... ).T
    >>> res = generate_hash_values(content, idx, num_perm, ngram_size, 0, hashranges, PERMUTATIONS)
    >>> len(res)
    10
    """
    hashvalues = np.ones(num_perm, dtype=np.uint64) * MAX_HASH
    tokens = list(map(str, tokens))
    ngrams = {" ".join(t) for t in get_ngrams(tokens, ngram_size, min_length)}
    hv = np.array([sha1_hash32(ngram.encode("utf-8")) for ngram in ngrams], dtype=np.uint64)
    a, b = permutations
    phv = np.bitwise_and(((hv * np.tile(a, (len(hv), 1)).T).T + b) % MERSENNE_PRIME, MAX_HASH)
    hashvalues = np.vstack([phv, hashvalues]).min(axis=0)
    Hs = [bytes(hashvalues[start:end].byteswap().data) for start, end in hashranges]
    return [(band_idx, H, idx) for band_idx, H in enumerate(Hs)]


def optimal_param(
        threshold: float,
        num_perm: int,
        false_positive_weight: float = 0.5,
        false_negative_weight: float = 0.5,
):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative, taken from datasketch.

    Parameters
    ----------
    threshold : float
        The threshold for similarity.
    num_perm : int
        The number of permutations.
    false_positive_weight : float
        The weight of false positive.
    false_negative_weight : float
        The weight of false negative.

    Returns
    -------
    Tuple[int, int]
        The optimal `b` and `r` parameters.
        The number of bands, and the number of rows per band respectively.

    Examples
    --------
    >>> optimal_param(0.7, 256)
    (25, 10)
    """

    def false_positive_area(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def area(s):
            return 1 - (1 - s ** float(r)) ** float(b)

        a, _ = integrate(area, 0.0, threshold)
        return a

    def false_negative_area(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def area(s):
            return 1 - (1 - (1 - s ** float(r)) ** float(b))

        a, _ = integrate(area, threshold, 1.0)
        return a

    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_area(threshold, b, r)
            fn = false_negative_area(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


def generate_edges(nodes: List[int]) -> List[Tuple[int, int]]:
    """
    Generate edges from a cluster. Instead of generating N^2 edges, we only need all nodes align to a single node, since
    we will be running connected components on the edges later.

    Parameters
    ----------
    nodes : List[int]
        The list of nodes in the cluster.

    Returns
    -------
    List[Tuple[int, int]]
        The list of edges.

    Examples
    --------
    >>> generate_edges([1, 2, 3])
    [(2, 1), (3, 1)]
    """
    if len(nodes) <= 1:
        return []

    min_node = min(nodes)
    return [(n, min_node) for n in nodes if n != min_node]


def fix_line_breaks(text: str):
    text = re.sub("(?<=[。；？：:\?”\)）!！])\s+(?=[\(（0-9\u4e00-\u9fa5])", r"\n", text).strip()
    return text


def fix_cite(text: str):
    regex_pattern = "(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]\s*$"
    if re.search(regex_pattern, text[-100:]):
        return ""
    else:
        return text


def fix_short_passage(text: str):
    if len(text) < 100:
        return ""
    else:
        return text


if __name__ == "__main__":  # pragma: no cover

    import argparse

    # pythia: applying near-deduplication with MinHashLSH and a threshold of 0.87
    # OPT: MinhashLSH with a Jaccard similarity ≥ .95.
    # Deduplicating Training Data Makes Language Models Better:
    # a collision at the desired Jaccard index threshold of 0.8 had a high probability of occurring.
    parser = argparse.ArgumentParser(description="Near-deduplicating Jsonl with PySpark")
    parser.add_argument("--input", type=str, required=True, help="input file to deduplicate")
    parser.add_argument("--file-format", type=str, default="jsonl")
    parser.add_argument("--threshold", type=float, default=0.87, help="Similarity threshold")
    parser.add_argument("--ngram_size", type=int, default=5, help="N-gram size")
    parser.add_argument("--min_length", type=int, default=5, help="Minimum length of document to be considered")
    parser.add_argument("--num_perm", type=int, default=256, help="Number of permutations")
    parser.add_argument("--b", type=int, default=None, help="Number of bands")
    parser.add_argument("--r", type=int, default=None, help="Number of rows per band")
    parser.add_argument("--column", "-c", type=str, default="content", help="Column to deduplicate")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--tokenize", action="store_true")
    parser.add_argument("--fixtext", action="store_true")
    parser.add_argument("--fix-line-break", action="store_true")
    parser.add_argument("--fix-cite", action="store_true")
    parser.add_argument("--fix-short-passage", action="store_true")
    args = parser.parse_args()

    conf = SparkConf()
    conf.set("spark.sql.hive.convertMetastoreParquet.mergeSchema", "false")
    conf.set("parquet.enable.summary-metadata", "false")
    conf.set("spark.app.name", "TextProcess")
    conf.set("spark.debug.maxToStringFields", "100")
    conf.set("spark.eventLog.enabled", "true")
    conf.set("spark.executor.memory", "456G")
    conf.set("spark.memory.fraction", "0.6")
    # conf.set("spark.shuffle.file.buffer", "1M")
    conf.set("spark.sql.shuffle.partitions", "600")
    # conf.set("spark.sql.shuffle.partitions", "200")
    conf.set("spark.sql.broadcastTimeout", "600")
    conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")  # 表示禁用
    conf.set("spark.dynamicAllocation.enabled", "true")
    conf.set("spark.sql.adaptive.enabled", "true")
    conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
    conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    conf.set("spark.sql.adaptive.localShuffleReader.enabled", "true")
    # conf.set("spark.file.transferTo", "false")
    # conf.set("spark.shuffle.unsafe.file.output.buffer", "1M")
    # conf.set("spark.shuffle.mapStatus.compression.codec", "lz4")
    # conf.set("spark.io.compression.lz4.blocksize", "512K")
    conf.set("spark.network.timeout", "10000000")
    conf.set("spark.shuffle.registration.timeout", "10000000")
    conf.set("spark.executor.cores", "100")
    conf.set("spark.shuffle.service.enabled", "false")
    # conf.set("spark.cores.max", 960)
    conf.set("spark.history.fs.logDirectory", "file:///cwwu/test_spark_data/spark_events")
    conf.set("spark.eventLog.dir", "file:///cwwu/test_spark_data/spark_events")
    conf.set("spark.executor.allowSparkContext", "true")
    conf.set("spark.driver.maxResultSize", "5120m")
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    log: Logger = spark.sparkContext._jvm.org.apache.log4j.LogManager.getLogger(__name__)  # type: ignore

    log.info(f"spark config: {spark.sparkContext.getConf().getAll()}")

    if args.b is None or args.r is None:
        B, R = optimal_param(args.threshold, args.num_perm)
        log.info(f"Using optimal parameters: {B=}, {R=}")
    else:
        B, R = args.b, args.r

    HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]
    PERMUTATIONS = np.array(
        [
            (
                RNG.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                RNG.randint(0, MERSENNE_PRIME, dtype=np.uint64),
            )
            for _ in range(args.num_perm)
        ],
        dtype=np.uint64,
    ).T

    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/cwwu/models/BAAI_AquilaChat-7B")
    # todo 可优化
    bc_tokenizer = spark.sparkContext.broadcast(tokenizer)
    tokenize_udf = F.udf(bc_tokenize, returnType=ArrayType(IntegerType()))

    fix_text_udf = F.udf(lambda x: ftfy.fix_text_segment(x).strip() if x else "")
    fix_line_breaks_udf = F.udf(lambda x: fix_line_breaks(x))
    fix_cite_udf = F.udf(lambda x: fix_cite(x))
    fix_short_passage_udf = F.udf(lambda x: fix_short_passage(x))
    log.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!start reading for local files.....")

    if args.file_format == "jsonl":
        df = spark.read.json(args.input)
    elif args.file_format == "parquet":
        df = spark.read.parquet(args.input)
    else:
        raise NotImplementedError()
    log.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!read data finished.....")
    # df.withColumn: Returns a new DataFrame by adding a column or replacing the existing column that has the same name.
    # make a new col with ids increase monotonically then Yields and caches the current DataFrame.

    if args.shuffle:
        print("do shuffle")
        shuf_seed = 2023
        df = df.orderBy(F.rand(seed=shuf_seed))
    if args.fixtext:
        df = df.withColumn(args.column, fix_text_udf(F.col(args.column)))
    if args.fix_line_break:
        df = df.withColumn(args.column, fix_line_breaks_udf(F.col(args.column)))
    if args.fix_cite:
        df = df.withColumn(args.column, fix_cite_udf(F.col(args.column)))
    if args.fix_short_passage:
        df = df.withColumn(args.column, fix_short_passage_udf(F.col(args.column)))
    # assert (args.deduplicate and args.tokenize) or not args.deduplicate, "dedup requires tokenization applied"
    # filter out empty rows
    df = df.filter(f'{args.column} != ""')
    # todo:删除input_ids字段
    if args.tokenize:
        df = df.withColumn("input_ids", tokenize_udf(F.col(args.column)))
    # df.show() # debug only
    # df.rdd: A Resilient Distributed Dataset (RDD), the basic abstraction in Spark.
    # Represents an immutable, partitioned collection of elements that can be operated on in parallel.
    # df.select: Projects a set of expressions and returns a new DataFrame.
    if args.deduplicate:
        log.info("start deduplication ...")
        df = df.withColumn("__id__", F.monotonically_increasing_id()).cache()
        records = df.select("__id__", "input_ids").rdd  # here we select two cols
        # rdd.repartition: Returns a new DataFrame partitioned by the given partitioning expressions.
        records = records.repartition(args.num_perm * 2).cache()

        # rdd.flatMap: Return a new RDD by first applying a function to all elements of this RDD, and then flattening the results.
        edges = (
            records.flatMap(
                lambda x: generate_hash_values(
                    tokens=x[1],  # gen hash with
                    idx=x[0],  # __id__
                    num_perm=args.num_perm,
                    ngram_size=args.ngram_size,
                    min_length=args.min_length,
                    hashranges=HASH_RANGES,
                    permutations=PERMUTATIONS,
                )
            )
            .groupBy(lambda x: (x[0], x[1]))
            .flatMap(lambda x: generate_edges([i[2] for i in x[1]]))
            .distinct()  # Returns a new DataFrame containing the distinct rows in this DataFrame.
            .cache()
        )

        a = edges
        while True:
            b = a.flatMap(large_star_map).groupByKey().flatMap(large_star_reduce).distinct().cache()
            a = b.map(small_star_map).groupByKey().flatMap(small_star_reduce).distinct().cache()
            changes = a.subtract(b).union(b.subtract(a)).collect()
            if len(changes) == 0:
                break

        results = a.collect()  # collect is an action

        # a, b, edges will not be used again
        try:
            a.unpersist()
            b.unpersist()
            edges.unpersist()
            print("successfully unpersist useless rdd")
        except Exception as e:
            print("skip unpersist due to: ", e)
            pass

        log.info("collect results succeed")
        if len(results) == 0:
            log.info("No duplicate components found.")
            df = df.drop("__id__").cache()
            df.write.parquet(args.output, mode="overwrite")
            sys.exit(0)

        components = spark.createDataFrame(results, schema=["__id__", "component"]).sort(["component", "__id__"])
        # components.show()
        df = df.join(components, on="__id__", how="left")
        df = df.filter(F.col("component").isNull()).drop("__id__", "component").cache()
        log.info("dedup finished...")
    log.info("write components to files")
    df.write.json(args.output, mode="overwrite")
    # df.write.parquet(args.output, mode="overwrite")
