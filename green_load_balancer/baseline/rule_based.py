def rule_based_decision(cpu, servers):
    if cpu > 80:
        return 1  # add server
    elif cpu < 30 and servers > 1:
        return 2  # remove server
    return 0  # do nothing