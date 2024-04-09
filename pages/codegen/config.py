import os

emb_model_name = "deepseek-coder-33b-instruct"
emb_model_path = os.path.join("/mnt/ssd2/model_store", emb_model_name)

index_name = 'velox-doc_deepseek-coder-33b-instruct'
vs_path = "/root/qyao/gitspace/udf_code_gen"
index_path = os.path.join(vs_path, index_name)

example_scala_code = """
@Description(
        name = "norm_str",
        value = "_FUNC_(input, [defaultValue], [dirtyValues ...]) trims input and " +
                "normalize null, empty or dirtyValues to defVal. \n",
        extended = "preset defaultValue is 'N-A' and preset dirtyValues are {'null', 'unknown', 'unknow', 'N-A'},\n" +
                   "the third NULL argument will clear the preset dirtyValues list."
)
public class UDFNormalizeString extends GenericUDF {


    public final static String DEFAULT_VALUE = "N-A";

    @SuppressWarnings("SpellCheckingInspection")
    public final static List<String> DEFAULT_NULL_VALUES = Arrays.asList("null", "unknown", "unknow", DEFAULT_VALUE);

    private transient String defaultValue;
    private transient Set<String> nullValues;

    @Override
    public ObjectInspector initialize(ObjectInspector[] arguments) throws UDFArgumentException {

        if (arguments.length == 0) {
            throw new UDFArgumentLengthException("norm_str() expects at least one argument.");
        }

        defaultValue = DEFAULT_VALUE;
        if (arguments.length >= 2) {

            // ............
            if (!ObjectInspectorUtils.isConstantObjectInspector(arguments[1])) {
                throw new UDFArgumentTypeException(1, "norm_str() expects a constant value as default.");
            }

            // .....
            Object writable = ObjectInspectorUtils.getWritableConstantValue(arguments[1]);
            defaultValue = (writable == null ? null : writable.toString());
        }

        nullValues = new HashSet<>(DEFAULT_NULL_VALUES);
        for (int i = 2; i < arguments.length; i++) {

            if (!ObjectInspectorUtils.isConstantObjectInspector(arguments[i])) {
                throw new UDFArgumentTypeException(i, "norm_str() expects constant values as dirty values");
            }

            Object writable = ObjectInspectorUtils.getWritableConstantValue(arguments[i]);

            if (writable == null) {
                // .........null .......
                if (i != 2) {
                    throw new UDFArgumentException(
                            "Only the third null argument will clear the default null values of norm_str().");
                }
                nullValues.clear();
            } else {
                nullValues.add(writable.toString().trim().toLowerCase());
            }
        }

        return PrimitiveObjectInspectorFactory.javaStringObjectInspector;
    }

    @Override
    public Object evaluate(DeferredObject[] arguments) throws HiveException {
        assert arguments.length > 0;

        Object inputObject = arguments[0].get();

        if (inputObject == null) {
            return defaultValue;
        }

        String input = inputObject.toString().trim();

        if (input.length() == 0 || nullValues.contains(input.toLowerCase())) {
            return defaultValue;
        }

        return input;
    }

    @Override
    public String getDisplayString(String[] children) {
        return getStandardDisplayString("norm_str", children);
    }
}

"""

demo_sample_python_code = """
@Description(
        name = "norm_str",
        value = "_FUNC_(input, [defaultValue], [dirtyValues ...]) trims input and " +
                "normalize null, empty or dirtyValues to defVal. \n",
        extended = "preset defaultValue is 'N-A' and preset dirtyValues are {'null', 'unknown', 'unknow', 'N-A'},\n" +
                   "the third NULL argument will clear the preset dirtyValues list."
)
public class UDFNormalizeString extends GenericUDF {


    public final static String DEFAULT_VALUE = "N-A";

    @SuppressWarnings("SpellCheckingInspection")
    public final static List<String> DEFAULT_NULL_VALUES = Arrays.asList("null", "unknown", "unknow", DEFAULT_VALUE);

    private transient String defaultValue;
    private transient Set<String> nullValues;

    @Override
    public ObjectInspector initialize(ObjectInspector[] arguments) throws UDFArgumentException {

        if (arguments.length == 0) {
            throw new UDFArgumentLengthException("norm_str() expects at least one argument.");
        }

        defaultValue = DEFAULT_VALUE;
        if (arguments.length >= 2) {

            // ............
            if (!ObjectInspectorUtils.isConstantObjectInspector(arguments[1])) {
                throw new UDFArgumentTypeException(1, "norm_str() expects a constant value as default.");
            }

            // .....
            Object writable = ObjectInspectorUtils.getWritableConstantValue(arguments[1]);
            defaultValue = (writable == null ? null : writable.toString());
        }

        nullValues = new HashSet<>(DEFAULT_NULL_VALUES);
        for (int i = 2; i < arguments.length; i++) {

            if (!ObjectInspectorUtils.isConstantObjectInspector(arguments[i])) {
                throw new UDFArgumentTypeException(i, "norm_str() expects constant values as dirty values");
            }

            Object writable = ObjectInspectorUtils.getWritableConstantValue(arguments[i]);

            if (writable == null) {
                // .........null .......
                if (i != 2) {
                    throw new UDFArgumentException(
                            "Only the third null argument will clear the default null values of norm_str().");
                }
                nullValues.clear();
            } else {
                nullValues.add(writable.toString().trim().toLowerCase());
            }
        }

        return PrimitiveObjectInspectorFactory.javaStringObjectInspector;
    }

    @Override
    public Object evaluate(DeferredObject[] arguments) throws HiveException {
        assert arguments.length > 0;

        Object inputObject = arguments[0].get();

        if (inputObject == null) {
            return defaultValue;
        }

        String input = inputObject.toString().trim();

        if (input.length() == 0 || nullValues.contains(input.toLowerCase())) {
            return defaultValue;
        }

        return input;
    }

    @Override
    public String getDisplayString(String[] children) {
        return getStandardDisplayString("norm_str", children);
    }
}
"""
