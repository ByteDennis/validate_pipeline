def case_insensitive_apply(stringA: str, stringB: str, func: Callable[[str, str], Any]) -> Any:
    """
    Apply a function to two strings after converting them to lowercase.
    
    Args:
        stringA: First string
        stringB: Second string
        func: Function that takes two strings and returns a result
        
    Returns:
        Result of applying func to lowercase versions of the strings
        
    Examples:
        >>> case_insensitive_apply("Hello", "HELLO", lambda a, b: a == b)
        True
        >>> case_insensitive_apply("ADd", "ad", lambda a, b: a.startswith(b))
        True
        >>> case_insensitive_apply("Python", "THON", lambda a, b: b in a)
        True
    """
    return func(stringA.lower(), stringB.lower())