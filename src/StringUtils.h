/*
 * StringUtils.h
 *
 *  Created on: Oct 15, 2012
 *      Author: Michal Busta
 */

#ifndef FASTTEXT_SRC_STRINGUTILS_H_
#define FASTTEXT_SRC_STRINGUTILS_H_

#include <string>
#include <vector>
#include <sstream>

#include <opencv2/core/core.hpp> // MIN/MAX

#include "utf8.h"

namespace cmp
{

template <typename T> std::string to_str(const T& t) { std::ostringstream os; os<<t; return os.str(); }

/**
 * @class cmp::StringUtils
 * 
 * @brief TODO brief description
 *
 * TODO type description
 */
class StringUtils
{
public:

	/**
	 * Trims input str
	 *
	 * @param src the source string
	 * @param c the characters to trim
	 * @return trimmed string
	 */
	inline static std::string trimStr(const std::string& src, const std::string& c = " \r\n")
	{
		size_t p2 = src.find_last_not_of(c);
		if (p2 == std::string::npos) return std::string();
		size_t p1 = src.find_first_not_of(c);
		if (p1 == std::string::npos) p1 = 0;
		return src.substr(p1, (p2-p1)+1);
	}

	static inline void split(std::vector<std::string>& lst, const std::string& input, const std::string& separators, bool remove_empty = true)
	{
	    std::ostringstream word;
	    for (size_t n = 0; n < input.size(); ++n)
	    {
	        if (std::string::npos == separators.find(input[n]))
	            word << input[n];
	        else
	        {
	            if (!word.str().empty() || !remove_empty)
	                lst.push_back(word.str());
	            word.str("");
	        }
	    }
	    if (!word.str().empty() || !remove_empty)
	        lst.push_back(word.str());
	}

	// convert UTF-8 string to wstring
	static inline std::wstring utf8_to_wstring (const std::string& str)
	{
		std::wstring wstr;
		utf8::utf8to32(str.begin(), str.end(), std::back_inserter(wstr));
		return wstr;
	}

	static inline std::string encode_utf8(const std::wstring& wstr)
	{
		std::string bytes;
	    utf8::utf32to8(wstr.begin(), wstr.end(), std::back_inserter(bytes));
	    return bytes;
	}

	static inline void add_utf8(const wchar_t wstr, std::string& bytes)
	{
		utf8::append(wstr,std::back_inserter(bytes));
	}

	static bool endsWith (std::string const &fullString, std::string const &ending)
	{
	    if (fullString.length() >= ending.length()) {
	        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
	    } else {
	        return false;
	    }
	}

	template <class gstring>
	inline static int levenshteinDistance (const gstring& source, const gstring& target)
	{
		// Step 1
		const int n = source.length();
		const int m = target.length();
		if (n == 0) {
			return m;
		}
		if (m == 0) {
			return n;
		}

		// Good form to declare a TYPEDEF

		typedef std::vector< std::vector<int> > Tmatrix;

		Tmatrix matrix(n+1);

		// Size the vectors in the 2.nd dimension. Unfortunately C++ doesn't
		// allow for allocation on declaration of 2.nd dimension of vec of vec

		for (int i = 0; i <= n; i++) {
			matrix[i].resize(m+1);
		}

		// Step 2

		for (int i = 0; i <= n; i++) {
			matrix[i][0]=i;
		}

		for (int j = 0; j <= m; j++) {
			matrix[0][j]=j;
		}

		// Step 3

		for (int i = 1; i <= n; i++) {

			const char s_i = source[i-1];

			// Step 4

			for (int j = 1; j <= m; j++) {

				const char t_j = target[j-1];

				// Step 5

				int cost;
				if (s_i == t_j) {
					cost = 0;
				}
				else {
					cost = 1;
				}

				// Step 6

				const int above = matrix[i-1][j];
				const int left = matrix[i][j-1];
				const int diag = matrix[i-1][j-1];
				int cell = MIN( above + 1, MIN(left + 1, diag + cost));

				// Step 6A: Cover transposition, in addition to deletion,
				// insertion and substitution. This step is taken from:
				// Berghel, Hal ; Roach, David : "An Extension of Ukkonen's
				// Enhanced Dynamic Programming ASM Algorithm"
				// (http://www.acm.org/~hlb/publications/asm/asm.html)

				if (i>2 && j>2) {
					int trans=matrix[i-2][j-2]+1;
					if (source[i-2]!=t_j) trans++;
					if (s_i!=target[j-2]) trans++;
					if (cell>trans) cell=trans;
				}

				matrix[i][j]=cell;
			}
		}
		// Step 7
		return matrix[n][m];
	}

};

} /* namespace cmp */
#endif /* FASTTEXT_SRC_STRINGUTILS_H_ */
