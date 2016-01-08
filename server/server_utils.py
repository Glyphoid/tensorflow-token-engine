def read_properties_file(properties_file_path):
    props = {}

    with open(properties_file_path, "rt") as f:
        for line in f:
            line = line.rstrip()

            # Remove the comment part
            if "#" in line:
                line = line[:line.index("#")]

            if len(line) == 0:
                continue

            if line.count("=") == 0:
                raise ValueError("Unable to parse properties file line: \"%s\"" % line)

            name = line[: line.index("=")]
            val = line[line.index("=") + 1 :]

            if name in props:
                raise ValueError("Duplicate property name: \"%s\"" % name)

            props[name] = val

    return props