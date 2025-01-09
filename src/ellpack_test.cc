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

#include "ellpack.h"

#include <gtest/gtest.h>

TEST(EllpackTest, set_1_value) {
  Ellpack matrix(10, 10, 3);
  int err = matrix.insert(0, 0, 1.0);
  EXPECT_EQ(err, 0);

  double value;
  bool found = matrix.get(value, 0, 0);

  EXPECT_TRUE(found);
  EXPECT_DOUBLE_EQ(value, 1.0);
}

TEST(EllpackTest, set_2_values) {
  Ellpack matrix(10, 10, 3);
  int err = matrix.insert(0, 0, 1.0);
  EXPECT_EQ(err, 0);
  err = matrix.insert(0, 1, 2.0);
  EXPECT_EQ(err, 0);

  {
    double value;
    bool found = matrix.get(value, 0, 0);
    EXPECT_TRUE(found);
    EXPECT_DOUBLE_EQ(value, 1.0);
  }
  {
    double value;
    bool found = matrix.get(value, 0, 1);
    EXPECT_TRUE(found);
    EXPECT_DOUBLE_EQ(value, 2.0);
  }
}

TEST(EllpackTest, value_not_found) {
  Ellpack matrix(10, 10, 3);

  double value;
  bool found = matrix.get(value, 0, 0);

  EXPECT_FALSE(found);
}

TEST(EllpackTest, insert_more_than_non_zeros_per_row) {
  Ellpack matrix(10, 10, 3);
  int err = matrix.insert(0, 0, 1.0);
  EXPECT_EQ(err, 0);
  err = matrix.insert(0, 1, 2.0);
  EXPECT_EQ(err, 0);
  err = matrix.insert(0, 2, 3.0);
  EXPECT_EQ(err, 0);
  err = matrix.insert(0, 3, 4.0);
  EXPECT_EQ(err, 1);

  double value;
  bool found = matrix.get(value, 0, 3);
  EXPECT_FALSE(found);
}

TEST(EllpackTest, insert_and_replace) {
  Ellpack matrix(10, 10, 3);
  int err = matrix.insert(0, 0, 1.0);
  EXPECT_EQ(err, 0);
  err = matrix.insert(0, 0, 2.0);
  EXPECT_EQ(err, 0);

  double value;
  bool found = matrix.get(value, 0, 0);

  EXPECT_TRUE(found);
  EXPECT_DOUBLE_EQ(value, 2.0);
}

TEST(EllpackTest, add) {
  // this is useful for finite elements
  Ellpack matrix(10, 10, 3);
  int err = matrix.add(0, 0, 1.0);
  EXPECT_EQ(err, 0);
  err = matrix.add(0, 0, 1.0);
  EXPECT_EQ(err, 0);

  double value;
  bool found = matrix.get(value, 0, 0);

  EXPECT_TRUE(found);
  EXPECT_DOUBLE_EQ(value, 2.0);
}

TEST(EllpackTest, delete_row) {
  Ellpack matrix(10, 10, 3);
  int err = matrix.insert(0, 0, 1.0);
  EXPECT_EQ(err, 0);
  err = matrix.insert(0, 1, 2.0);
  EXPECT_EQ(err, 0);
  err = matrix.insert(0, 2, 3.0);
  EXPECT_EQ(err, 0);

  matrix.deleteRow(0);

  double value;

  bool found = matrix.get(value, 0, 0);
  EXPECT_FALSE(found);
  found = matrix.get(value, 0, 1);
  EXPECT_FALSE(found);
  found = matrix.get(value, 0, 2);
  EXPECT_FALSE(found);
}
