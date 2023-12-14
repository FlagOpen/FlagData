from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, array
from pyspark.sql.types import ArrayType, StringType

# Initialize Spark session
spark = SparkSession.builder.appName("StringMatchingUDF").getOrCreate()


# Function to string matching
def stringMatching(words):
    ans = []
    for i, x in enumerate(words):
        for j, y in enumerate(words):
            if j != i and x in y:
                ans.append(x)
                break
    return ans


# Register the function as a UDF
string_matching_udf = udf(stringMatching, ArrayType(StringType()))

# Sample input data
data = [("blue", "green", "bu")]

# Create a DataFrame
df = spark.createDataFrame(data, ["word1", "word2", "word3"])
# Dynamically build arrays without hard-coding the number of columns
# df = spark.createDataFrame(data, ["words"])

# Apply the UDF to the DataFrame
result_df = df.withColumn("matched_words", string_matching_udf(array("word1", "word2", "word3")))
# Dynamically build arrays without hard-coding the number of columns
# result_df = df.withColumn("matched_words", string_matching_udf(array(*df.columns)))

# Show the result
result_df.show(truncate=False)

# Stop Spark session
spark.stop()

"""
很多时候，我们想使用spark的分布式处理数据能力，这里提供了一个普通函数改造成spark udf函数，进而使用spark能力的方法，
但是对于想要改造成spark任务的函数需要满足：
1、 数据并行性：函数的输入数据可以划分为多个部分并进行并行处理。
2、可序列化和不可变性：Spark 中的函数必须是可序列化的，以便在不同节点上传输。
3、不依赖于特定计算节点：函数的执行不依赖于特定节点的计算资源或数据存储位置，以便能够在集群中的任何节点上执行。
4、无状态或可共享状态：函数不依赖于外部状态或只依赖于可共享的状态。这样可以确保在不同计算节点上并行执行函数时不会发生冲突或竞争条件。


 在使用 UDF 时，应该考虑性能和优化。有些函数可能在本地 Python 环境中运行良好，但在分布式 Spark 环境中可能效率不高。
 对于复杂的逻辑或需要大量内存的函数，可能需要进一步的优化和考虑。UDF 是为了简单的逻辑和数据处理而设计的，对于更复杂的计算，可能需要使用 Spark 的原生算子来进行处理。

"""
