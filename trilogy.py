def solution(crypt):
    # Extract unique letters
    letters = set(''.join(crypt))
    if len(letters) > 10:
        return 0
    
    # Precompute leading letters to avoid repeated checks
    leading_letters = {word[0] for word in crypt if len(word) > 1}
    
    # Convert letters to list for consistent ordering
    letter_list = list(letters)
    
    solutions = 0
    
    def backtrack(index, used_digits, mapping):
        nonlocal solutions
        
        # Base case: mapped all letters
        if index == len(letter_list):
            # Convert words to numbers
            nums = []
            for word in crypt:
                # Check for leading zeros
                if len(word) > 1 and mapping[word[0]] == 0:
                    return
                
                # Convert word to number
                num = int(''.join(str(mapping[c]) for c in word))
                nums.append(num)
            
            # Check if equation is valid
            if nums[0] + nums[1] == nums[2]:
                solutions += 1
            return
        
        # Current letter to map
        current_letter = letter_list[index]
        
        # Skip if leading letter can't be zero
        start = 0 if current_letter not in leading_letters else 1
        
        # Try digits
        for digit in range(start, 10):
            # Check if digit is already used
            if used_digits & (1 << digit):
                continue
            
            # Map letter and continue
            mapping[current_letter] = digit
            backtrack(index + 1, used_digits | (1 << digit), mapping)
            
            # Backtrack
            del mapping[current_letter]
    
    # Start backtracking
    backtrack(0, 0, {})
    
    return solutions
print(solution(["SEND", "MORE", "MONEY"]))