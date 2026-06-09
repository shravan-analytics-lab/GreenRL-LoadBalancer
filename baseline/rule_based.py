def rule_based_decision(cpu, servers, queue=0):
    # Conservative autoscaling baseline: stable, but often keeps extra capacity.
    if cpu > 65 or queue > 40:
        return 1

    if cpu < 35 and queue == 0 and servers > 2:
        return 2

    return 0
