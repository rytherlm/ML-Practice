def is_prime(num):
    if num <= 1:
        return False
    if num == 2:
        return True
    if num % 2 == 0:
        return False
    for i in range(3, int(num**0.5) + 1, 2):
        if num % i == 0:
            return False
    return True

def write_primes_to_file(limit, filename):
    with open(filename, 'w') as file:
        for i in range(1, limit + 1):
            if is_prime(i):
                file.write(f"{i}\n")

# Example usage:
limit = 20000
write_primes_to_file(limit, "prime.txt")
