/*
 *  This source code is part of SparseX: a library to perform operations with
 *  sparse matrices.
 *
 *  Copyright (C) - 2025 - Guido Giuntoli <gagiuntoli@gmail.com>
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published
 *  by the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef SPARSEX_ALGEBRA_H
#define SPARSEX_ALGEBRA_H

#include <array>
#include <cstdlib>
#include <vector>

double dot(const std::vector<double> &y, const std::vector<double> &x, size_t n);
double norm(const std::vector<double> &x, size_t n);

#endif
