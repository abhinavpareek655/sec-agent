from urllib.parse import urlparse


def is_host_allowed(host: str, allowed_hosts: set[str]) -> bool:
    """
    Check if a hostname is in the allowed hosts set.
    
    Args:
        host: The hostname or IP to check
        allowed_hosts: Set of allowed hostnames (normalized to lowercase)
        
    Returns:
        True if the host is allowed, False otherwise
    """
    if not host:
        return False
    return host.lower() in allowed_hosts


def is_target_valid(target: str) -> bool:
    """
    Validate a target string to prevent argument injection.
    
    Rejects targets that:
    - Start with '-' (looks like an nmap flag)
    - Contain shell metacharacters
    
    Args:
        target: The target string to validate
        
    Returns:
        True if the target looks safe, False otherwise
    """
    if not target or target.startswith("-"):
        return False
    
    # Reject common shell metacharacters
    dangerous_chars = {",","$", "`", ";", "&", "|", "<", ">", "(", ")", "{", "}", "\n", "\r"}
    if any(char in target for char in dangerous_chars):
        return False
    
    return True
